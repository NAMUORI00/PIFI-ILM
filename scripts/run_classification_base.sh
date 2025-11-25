#!/usr/bin/env bash
set -euo pipefail

# Legacy-style classification BASE experiment (root-path layout)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# Pick the emptiest GPU if not specified
choose_free_gpu() {
  nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | sort -n | head -1 | awk -F, '{print $2}'
}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(choose_free_gpu)"
fi
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb tweet_sentiment_binary tweet_offensive cola"}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
USE_WANDB=${USE_WANDB:-false}
USE_TENSORBOARD=${USE_TENSORBOARD:-false}

CACHE_PATH=${CACHE_PATH:-"$ROOT_DIR/cache"}
PREPROCESS_PATH=${PREPROCESS_PATH:-"$ROOT_DIR/preprocessed"}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$ROOT_DIR/checkpoints"}
RESULT_PATH=${RESULT_PATH:-"$ROOT_DIR/results"}
LOG_PATH=${LOG_PATH:-"$ROOT_DIR/tensorboard_logs"}

COMMON_ARGS=(
  --model_type "$MODEL"
  --cache_path "$CACHE_PATH"
  --preprocess_path "$PREPROCESS_PATH"
  --model_path "$MODEL_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --result_path "$RESULT_PATH"
  --log_path "$LOG_PATH"
)

for DS in ${DATASETS}; do
  echo "[BASE] Dataset=${DS} | Model=${MODEL}"

  # 1) Preprocess
  python main.py --task classification --job preprocessing --task_dataset "$DS" \
    "${COMMON_ARGS[@]}" --seed "$SEED"

  # 2) BASE train/test
  python main.py --task classification --job training --task_dataset "$DS" --test_dataset "$DS" --method base \
    --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"

  python main.py --task classification --job testing --task_dataset "$DS" --test_dataset "$DS" --method base \
    --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"
done

echo "[BASE] Completed classification baseline experiments."
