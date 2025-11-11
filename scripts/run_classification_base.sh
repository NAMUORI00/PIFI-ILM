#!/usr/bin/env bash
set -euo pipefail

# Legacy-style classification BASE experiment (root-path layout)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb tweet_sentiment_binary tweet_offensive cola"}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
USE_WANDB=${USE_WANDB:-false}
USE_TENSORBOARD=${USE_TENSORBOARD:-false}

COMMON_ARGS=(
  --model_type "$MODEL"
  --cache_path "$ROOT_DIR/cache"
  --preprocess_path "$ROOT_DIR/preprocessed"
  --model_path "$ROOT_DIR/models"
  --checkpoint_path "$ROOT_DIR/checkpoints"
  --result_path "$ROOT_DIR/results"
  --log_path "$ROOT_DIR/tensorboard_logs"
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

