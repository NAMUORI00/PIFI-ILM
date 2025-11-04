#!/usr/bin/env bash
set -euo pipefail

# Run original classification experiments with PiFi fixed to the LAST LLM layer (no ILM selection).
# Env overrides:
#   LLM (default: qwen2_0.5b)
#   MODEL (default: bert)
#   DATASETS (default: "sst2 imdb tweet_sentiment_binary tweet_offensive cola")
#   EPOCHS, BS, WORKERS, SEED (defaults: 1, 8, 0, 2023)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR/Classification"

LLM=${LLM:-qwen2_0.5b}
MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb tweet_sentiment_binary tweet_offensive cola"}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}

COMMON_ARGS=(
  --model_type "$MODEL"
  --preprocess_path ./preprocessed
  --cache_path ./cache
  --data_path ./data
  --model_path ./models
  --checkpoint_path ./checkpoints
  --result_path ./results
  --log_path ./logs
)

for DS in ${DATASETS}; do
  echo "[LAST] Dataset=${DS} | Model=${MODEL} | LLM=${LLM}"

  # 1) Preprocess
  python main.py --task classification --job=preprocessing --task_dataset="$DS" "${COMMON_ARGS[@]}" --seed "$SEED"

  # 2) BASE train/test
  python main.py --task classification --job=training --task_dataset="$DS" --test_dataset="$DS" --method=base \
    --num_epochs="$EPOCHS" --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"
  python main.py --task classification --job=testing  --task_dataset="$DS" --test_dataset="$DS" --method=base \
    --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"

  # 3) PiFi LAST-layer train/test (no ILM selection)
  python main.py --task classification --job=training --task_dataset="$DS" --test_dataset="$DS" --method=pifi --llm "$LLM" \
    --auto_select_layer false --layer_num -1 \
    --num_epochs="$EPOCHS" --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"
  python main.py --task classification --job=testing  --task_dataset="$DS" --test_dataset="$DS" --method=pifi --llm "$LLM" \
    --auto_select_layer false --layer_num -1 \
    --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"
done

echo "[LAST] Completed classification experiments with LAST-layer PiFi."

