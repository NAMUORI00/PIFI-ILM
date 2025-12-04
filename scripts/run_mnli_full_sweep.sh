#!/usr/bin/env bash
set -euo pipefail

# Full layer sweep for MNLI entailment with PiFi
# Warning: very long runtime (24 layers * 3 epochs on full MNLI)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# Pick emptiest GPU if not specified
choose_free_gpu() {
  nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | sort -n | head -1 | awk -F, '{print $2}'
}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(choose_free_gpu)"
fi
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Config
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
LLM=${LLM:-qwen2_0.5b}

CACHE_PATH=${CACHE_PATH:-"$ROOT_DIR/cache"}
PREPROCESS_PATH=${PREPROCESS_PATH:-"$ROOT_DIR/preprocessed"}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$ROOT_DIR/checkpoints"}
RESULT_PATH=${RESULT_PATH:-"$ROOT_DIR/results"}

COMMON_ARGS=(
  --task entailment
  --task_dataset mnli
  --test_dataset mnli
  --method pifi
  --llm "$LLM"
  --num_epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$WORKERS"
  --use_wandb false
  --cache_path "$CACHE_PATH"
  --preprocess_path "$PREPROCESS_PATH"
  --model_path "$MODEL_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --result_path "$RESULT_PATH"
)

# 1) Preprocess once
python main.py --job preprocessing "${COMMON_ARGS[@]}" --seed "$SEED"

# 2) Sweep layers 0..23
for L in $(seq 0 23); do
  echo "[TRAIN] layer=$L"
  python main.py --job training "${COMMON_ARGS[@]}" --layer_num "$L" --seed "$SEED"
  echo "[TEST] layer=$L"
  python main.py --job testing  "${COMMON_ARGS[@]}" --layer_num "$L" --seed "$SEED"
done

echo "[DONE] MNLI full layer sweep completed."
