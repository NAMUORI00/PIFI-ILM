#!/usr/bin/env bash
set -euo pipefail

# Legacy-style classification PiFi LAST-layer experiment (root-path layout)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

LLM=${LLM:-qwen2_0.5b}
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
  echo "[LAST] Dataset=${DS} | Model=${MODEL} | LLM=${LLM}"

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

  # 3) PiFi LAST-layer train/test
  python main.py --task classification --job training --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
    --auto_select_layer false --layer_num -1 \
    --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"

  python main.py --task classification --job testing --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
    --auto_select_layer false --layer_num -1 \
    --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"
done

echo "[LAST] Completed classification PiFi LAST-layer experiments."

