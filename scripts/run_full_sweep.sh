#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PiFi Full Layer Sweep Script
# =============================================================================
# Runs training and testing for all layers (0-MAX_LAYER) on specified dataset(s).
#
# Usage:
#   TASK=classification DATASET=imdb ./run_full_sweep.sh
#   TASK=entailment DATASET=mnli ./run_full_sweep.sh
#   TASK=classification DATASET="sst2 cola imdb" ./run_full_sweep.sh  # Multiple datasets
#
# See scripts/manual.md for full documentation
# =============================================================================

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# === GPU Selection ===
choose_free_gpu() {
  nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 | awk -F, '{print $2}' || echo "0"
}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(choose_free_gpu)"
fi
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# === Configuration ===
TASK=${TASK:-classification}
DATASET=${DATASET:-sst2}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
LLM=${LLM:-qwen2_0.5b}
MAX_LAYER=${MAX_LAYER:-23}
USE_WANDB=${USE_WANDB:-false}

# === Paths ===
CACHE_PATH=${CACHE_PATH:-"$ROOT_DIR/cache"}
PREPROCESS_PATH=${PREPROCESS_PATH:-"$ROOT_DIR/preprocessed"}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$ROOT_DIR/checkpoints"}
RESULT_PATH=${RESULT_PATH:-"$ROOT_DIR/results"}

echo "[INFO] Task: $TASK"
echo "[INFO] Dataset(s): $DATASET"
echo "[INFO] Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "[INFO] LLM: $LLM, Max Layer: $MAX_LAYER"
echo "[INFO] W&B Logging: $USE_WANDB"

for DS in $DATASET; do
  echo ""
  echo "===== Dataset: $DS ====="

  COMMON_ARGS=(
    --task "$TASK"
    --task_dataset "$DS"
    --test_dataset "$DS"
    --method pifi
    --llm "$LLM"
    --num_epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$WORKERS"
    --use_wandb "$USE_WANDB"
    --cache_path "$CACHE_PATH"
    --preprocess_path "$PREPROCESS_PATH"
    --model_path "$MODEL_PATH"
    --checkpoint_path "$CHECKPOINT_PATH"
    --result_path "$RESULT_PATH"
  )

  # 1) Preprocess once
  echo "[PREPROCESS] $DS"
  python main.py --job preprocessing "${COMMON_ARGS[@]}" --seed "$SEED"

  # 2) Sweep layers 0..MAX_LAYER
  for L in $(seq 0 "$MAX_LAYER"); do
    echo "[TRAIN] $DS layer=$L"
    python main.py --job training "${COMMON_ARGS[@]}" --layer_num "$L" --seed "$SEED"
    echo "[TEST] $DS layer=$L"
    python main.py --job testing "${COMMON_ARGS[@]}" --layer_num "$L" --seed "$SEED"
  done

  echo "[DONE] $DS full layer sweep completed."
done

echo ""
echo "[DONE] All datasets completed: $DATASET"
