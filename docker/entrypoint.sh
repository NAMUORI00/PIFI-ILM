#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/app}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# Ensure artifact directories exist (ignore errors on bind mounts)
for d in cache preprocessed models checkpoints results tensorboard_logs wandb; do
  mkdir -p "$ROOT_DIR/$d" || true
done

# Prefer writable HF cache inside /app/cache if not set
export HF_HOME=${HF_HOME:-$ROOT_DIR/cache/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME}

# First-run auto experiment (classification + ILM)
FIRST_RUN=${FIRST_RUN:-true}
FIRST_MARKER="$ROOT_DIR/results/.first_run_done"
if [[ "$FIRST_RUN" == "true" && ! -f "$FIRST_MARKER" ]]; then
  echo "[entrypoint] First run detected. Launching classification+ILM quick experiment..."
  export DATASETS=${DATASETS:-sst2}
  export MODEL=${MODEL:-bert}
  export LLM=${LLM:-llama3.1}
  export EPOCHS=${EPOCHS:-1}
  export BS=${BS:-8}
  export WORKERS=${WORKERS:-2}
  export USE_WANDB=${USE_WANDB:-false}
  export USE_TENSORBOARD=${USE_TENSORBOARD:-false}

  if [[ -x "$ROOT_DIR/scripts/run_classification_pifi_ilm.sh" ]]; then
    bash "$ROOT_DIR/scripts/run_classification_pifi_ilm.sh" || echo "[entrypoint] ILM runner returned non-zero"
  else
    echo "[entrypoint] Fallback to direct main.py call"
    python "$ROOT_DIR/main.py" \
      --task classification --job preprocessing --task_dataset "${DATASETS}" \
      --cache_path  "$ROOT_DIR/cache" \
      --preprocess_path "$ROOT_DIR/preprocessed" \
      --model_path  "$ROOT_DIR/models" \
      --checkpoint_path "$ROOT_DIR/checkpoints" \
      --result_path "$ROOT_DIR/results" \
      --log_path "$ROOT_DIR/tensorboard_logs" \
      --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" || true

    python "$ROOT_DIR/main.py" \
      --task classification --job training --task_dataset "${DATASETS}" --method pifi --llm_model "$LLM" \
      --auto_select_layer true --selection_samples 400 --selection_pcs 16 --selection_top_pc 5 \
      --selection_pooling mean --selection_dtype fp16 --selection_max_length 128 \
      --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
      --cache_path  "$ROOT_DIR/cache" \
      --preprocess_path "$ROOT_DIR/preprocessed" \
      --model_path  "$ROOT_DIR/models" \
      --checkpoint_path "$ROOT_DIR/checkpoints" \
      --result_path "$ROOT_DIR/results" \
      --log_path "$ROOT_DIR/tensorboard_logs" \
      --use_wandb "$USE_WANDB" --use_tensorboard "$USE_TENSORBOARD" || true
  fi

  mkdir -p "$(dirname "$FIRST_MARKER")" && date > "$FIRST_MARKER" || true
  echo "[entrypoint] First run complete. Marker written to $FIRST_MARKER"
fi

# Continue with provided command or default shell
if [[ $# -gt 0 ]]; then
  exec "$@"
else
  exec bash
fi
