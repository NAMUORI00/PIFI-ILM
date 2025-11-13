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
docker compose build

echo "[docker] Starting service (skip entrypoint autorun)..."
FIRST_RUN=false docker compose up -d pifi

echo "[docker] Checking CUDA..."
docker compose exec -T pifi python - <<'PY'
import torch
ok = torch.cuda.is_available()
print('CUDA:', ok)
print('GPU:', torch.cuda.get_device_name(0) if ok else 'CPU')
assert ok, 'CUDA is not available in container'
PY

# Prefer writable HF cache under /app/cache (host-mapped)
HFVARS=( -e HF_HOME=/app/cache/hf )

echo "[step 1/3] Preprocessing: task=classification dataset=${DATASET}"
docker compose exec "${HFVARS[@]}" -T pifi \
  python main.py --task classification --job preprocessing --task_dataset "$DATASET" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN"

echo "[step 2/3] Training BASE: epochs=${EPOCHS} bs=${BS}"
docker compose exec "${HFVARS[@]}" -T pifi \
  python main.py --task classification --job training --task_dataset "$DATASET" --method base \
  --model_type "$MODEL" --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[step 3/3] Testing BASE"
docker compose exec "${HFVARS[@]}" -T pifi \
  python main.py --task classification --job testing --task_dataset "$DATASET" --method base \
  --model_type "$MODEL" --batch_size "$BS" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[done] Artifacts:"
echo "  - models/classification/${DATASET}/cls/${MODEL}/base/llama3.1/-1/final_model.pt"
echo "  - checkpoints/classification/${DATASET}/cls/${MODEL}/base/llama3.1/-1/checkpoint.pt"
echo "  - results/, tensorboard_logs/"

