#!/usr/bin/env bash
set -euo pipefail

# Run classification BASE experiment inside Docker (compose)
# - Preprocess -> Train (base) -> Test
# - Artifacts are preserved via compose volume mounts

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Config (override via env)
DATASET=${DATASET:-sst2}
MODEL=${MODEL:-bert}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
WORKERS=${WORKERS:-2}
MAX_LEN=${MAX_LEN:-64}
WANDB=${WANDB:-false}
TENSORBOARD=${TENSORBOARD:-false}

echo "[docker] Building image..."
docker build -t pifi:cu118 .

echo "[docker] Checking CUDA with docker run..."
docker run --rm --gpus all pifi:cu118 python - <<'PY'
import torch
ok = torch.cuda.is_available()
print('CUDA:', ok)
print('GPU:', torch.cuda.get_device_name(0) if ok else 'CPU')
assert ok, 'CUDA is not available in container'
PY

# Prefer writable HF cache under /app/cache (host-mapped)
HFVARS=( -e HF_HOME=/app/cache/hf -e FIRST_RUN=false )
VOLS=( \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/preprocessed:/app/preprocessed" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/tensorboard_logs:/app/tensorboard_logs" \
  -v "$(pwd)/wandb:/app/wandb" \
  -v "$(pwd)/.hf_cache:/opt/hf-cache" \
  -v "$(pwd)/dataset:/app/dataset" \
)

echo "[step 1/3] Preprocessing: task=classification dataset=${DATASET}"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job preprocessing --task_dataset "$DATASET" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN"

echo "[step 2/3] Training BASE: epochs=${EPOCHS} bs=${BS}"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job training --task_dataset "$DATASET" --method base \
  --model_type "$MODEL" --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[step 3/3] Testing BASE"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job testing --task_dataset "$DATASET" --method base \
  --model_type "$MODEL" --batch_size "$BS" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[done] Artifacts:"
echo "  - models/classification/${DATASET}/cls/${MODEL}/base/llama3.1/-1/final_model.pt"
echo "  - checkpoints/classification/${DATASET}/cls/${MODEL}/base/llama3.1/-1/checkpoint.pt"
echo "  - results/, tensorboard_logs/"
