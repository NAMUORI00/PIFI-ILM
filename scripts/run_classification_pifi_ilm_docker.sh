#!/usr/bin/env bash
set -euo pipefail

# Run classification PiFi + ILM experiment inside Docker (compose)
# - Preprocess -> Train (pifi + auto_select_layer) -> Test (using selected layer)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Config (override via env)
DATASET=${DATASET:-sst2}
MODEL=${MODEL:-bert}
LLM=${LLM:-llama3.1}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-2}
SEL_SAMPLES=${SEL_SAMPLES:-400}
SEL_PCS=${SEL_PCS:-16}
SEL_TOP_PC=${SEL_TOP_PC:-5}
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

echo "[step 1/4] Preprocessing: task=classification dataset=${DATASET}"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job preprocessing --task_dataset "$DATASET" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN"

echo "[step 2/4] Training BASE (for baseline comparison)"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job training --task_dataset "$DATASET" --method base \
  --model_type "$MODEL" --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" --max_seq_len "$MAX_LEN" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[step 3/4] Training PiFi with ILM auto-selection"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job training --task_dataset "$DATASET" --method pifi --llm_model "$LLM" \
  --auto_select_layer true --selection_samples "$SEL_SAMPLES" --selection_pcs "$SEL_PCS" --selection_top_pc "$SEL_TOP_PC" \
  --selection_pooling mean --selection_dtype fp16 --selection_max_length 128 \
  --model_type "$MODEL" --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

SEL_JSON="results/layer_selection/classification/${DATASET}/${MODEL}/${LLM}/selection.json"
echo "[info] Selection file path: $SEL_JSON"
LAYER=$(docker run --rm --gpus all -w /app "${VOLS[@]}" pifi:cu118 python - <<PY
import json,sys,os
p = '/app/${SEL_JSON}'
try:
    with open(p) as f:
        obj=json.load(f)
    print(int(obj.get('best_llm_layer', -1)))
except Exception as e:
    print(-1)
PY
)
echo "[info] Selected layer: ${LAYER}"

echo "[step 4/4] Testing PiFi with selected layer"
docker run --rm --gpus all -w /app "${HFVARS[@]}" "${VOLS[@]}" pifi:cu118 \
  python main.py --task classification --job testing --task_dataset "$DATASET" --method pifi --llm_model "$LLM" \
  --auto_select_layer false --layer_num "$LAYER" \
  --model_type "$MODEL" --batch_size "$BS" --num_workers "$WORKERS" \
  --use_wandb "$WANDB" --use_tensorboard "$TENSORBOARD"

echo "[done] Artifacts:"
echo "  - results/layer_selection/classification/${DATASET}/${MODEL}/${LLM}/selection.json"
echo "  - models/classification/${DATASET}/cls/${MODEL}/pifi/${LLM}/*/final_model.pt"
echo "  - checkpoints/..."
