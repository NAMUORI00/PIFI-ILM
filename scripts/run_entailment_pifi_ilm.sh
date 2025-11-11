#!/usr/bin/env bash
set -euo pipefail

# Legacy-style entailment PiFi with ILM auto-selection (root-path layout)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

LLM=${LLM:-llama3.1}
MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"mnli snli"}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-2023}
SEL_SAMPLES=${SEL_SAMPLES:-400}
SEL_PCS=${SEL_PCS:-16}
SEL_TOP_PC=${SEL_TOP_PC:-5}
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
  echo "[ILM] Dataset=${DS} | Model=${MODEL} | LLM=${LLM}"

  # 1) Preprocess
  python main.py --task entailment --job preprocessing --task_dataset "$DS" \
    "${COMMON_ARGS[@]}" --seed "$SEED"

  # 2) BASE train/test
  python main.py --task entailment --job training --task_dataset "$DS" --test_dataset "$DS" --method base \
    --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"
  python main.py --task entailment --job testing --task_dataset "$DS" --test_dataset "$DS" --method base \
    --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"

  # 3) PiFi train with ILM auto selection
  python main.py --task entailment --job training --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
    --auto_select_layer true --selection_samples "$SEL_SAMPLES" --selection_pcs "$SEL_PCS" --selection_top_pc "$SEL_TOP_PC" \
    --selection_pooling mean --selection_dtype fp16 --selection_max_length 128 \
    --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"

  # 4) PiFi test using the selected layer from selection.json
  SEL_JSON="$ROOT_DIR/results/layer_selection/entailment/${DS}/${MODEL}/${LLM}/selection.json"
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

  python main.py --task entailment --job testing --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
    --auto_select_layer false --layer_num "$LAYER" \
    --batch_size "$BS" --num_workers "$WORKERS" \
    --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" "${COMMON_ARGS[@]}"
done

echo "[ILM] Completed entailment PiFi ILM experiments."

