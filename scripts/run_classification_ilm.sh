#!/usr/bin/env bash
set -euo pipefail

# Run original classification experiments with ILM-based auto layer selection enabled for PiFi.
# Env overrides:
#   LLM (default: llama3.1)
#   MODEL (default: bert)
#   DATASETS (default: "sst2 imdb tweet_sentiment_binary tweet_offensive cola")
#   EPOCHS, BS, WORKERS, SEED (defaults: 1, 8, 0, 2023)
#   SEL_SAMPLES, SEL_PCS, SEL_TOP_PC (defaults: 400, 16, 5)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR/Classification"

LLM=${LLM:-llama3.1}
MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb tweet_sentiment_binary tweet_offensive cola"}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
SEL_SAMPLES=${SEL_SAMPLES:-400}
SEL_PCS=${SEL_PCS:-16}
SEL_TOP_PC=${SEL_TOP_PC:-5}

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
  echo "[ILM] Dataset=${DS} | Model=${MODEL} | LLM=${LLM}"

  # 1) Preprocess
  python main.py --task classification --job=preprocessing --task_dataset="$DS" "${COMMON_ARGS[@]}" --seed "$SEED"

  # 2) BASE train/test (original flow)
  python main.py --task classification --job=training --task_dataset="$DS" --test_dataset="$DS" --method=base \
    --num_epochs="$EPOCHS" --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"
  python main.py --task classification --job=testing  --task_dataset="$DS" --test_dataset="$DS" --method=base \
    --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"

  # 3) PiFi train with ILM auto selection (direct PC patching on LLM)
  python main.py --task classification --job=training --task_dataset="$DS" --test_dataset="$DS" --method=pifi --llm "$LLM" \
    --auto_select_layer true --selection_samples "$SEL_SAMPLES" --selection_pcs "$SEL_PCS" --selection_top_pc "$SEL_TOP_PC" \
    --selection_pooling mean --selection_dtype fp16 --selection_max_length 128 \
    --num_epochs="$EPOCHS" --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"

  # 4) PiFi test using the selected layer from selection.json
  SEL_JSON="./results/layer_selection/classification/${DS}/${MODEL}/${LLM}/selection.json"
  if [ -f "$SEL_JSON" ]; then
    echo "[ILM] Using selection file: $SEL_JSON"
    LAYER=$(python - <<PY
import json,sys
with open('$SEL_JSON') as f:
    obj=json.load(f)
print(int(obj.get('best_llm_layer', -1)))
PY
)
  else
    echo "[ILM] Selection file not found; defaulting to -1"
    LAYER=-1
  fi

  python main.py --task classification --job=testing  --task_dataset="$DS" --test_dataset="$DS" --method=pifi --llm "$LLM" \
    --auto_select_layer false --layer_num "$LAYER" \
    --batch_size="$BS" --num_workers="$WORKERS" --use_wandb false --use_tensorboard false "${COMMON_ARGS[@]}"
done

echo "[ILM] Completed classification experiments with ILM selection."
