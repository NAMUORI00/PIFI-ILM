#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PiFi Classification Unified Experiment Script
# =============================================================================
# Usage:
#   MODE=base ./run_classification.sh              # Base model only
#   MODE=pifi ./run_classification.sh              # PiFi with last layer
#   MODE=pifi_ilm ./run_classification.sh          # PiFi with ILM auto-selection
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

# === Experiment Mode ===
MODE=${MODE:-pifi_ilm}  # base | pifi | pifi_ilm

# === Common Settings ===
MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb cola"}
EPOCHS=${EPOCHS:-3}
BS=${BS:-32}
WORKERS=${WORKERS:-2}
SEED=${SEED:-2023}

# === PiFi Settings ===
LLM=${LLM:-llama3.1}
LAYER=${LAYER:--1}  # -1 means last layer (pifi) or auto-select (pifi_ilm)

# === ILM Selection Settings (pifi_ilm mode only) ===
SEL_SAMPLES=${SEL_SAMPLES:-400}
SEL_PCS=${SEL_PCS:-16}
SEL_TOP_PC=${SEL_TOP_PC:-5}

# === Environment Settings ===
DOCKER=${DOCKER:-false}
USE_WANDB=${USE_WANDB:-true}

# === Path Settings (auto-detect Docker vs bare metal) ===
if [[ "$DOCKER" == "true" ]] || [[ -f /.dockerenv ]]; then
  CACHE_PATH=${CACHE_PATH:-"/app/cache"}
  PREPROCESS_PATH=${PREPROCESS_PATH:-"/app/preprocessed"}
  MODEL_PATH=${MODEL_PATH:-"/app/models"}
  CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/app/checkpoints"}
  RESULT_PATH=${RESULT_PATH:-"/app/results"}
else
  CACHE_PATH=${CACHE_PATH:-"$ROOT_DIR/cache"}
  PREPROCESS_PATH=${PREPROCESS_PATH:-"$ROOT_DIR/preprocessed"}
  MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models"}
  CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$ROOT_DIR/checkpoints"}
  RESULT_PATH=${RESULT_PATH:-"$ROOT_DIR/results"}
fi

COMMON_ARGS=(
  --model_type "$MODEL"
  --cache_path "$CACHE_PATH"
  --preprocess_path "$PREPROCESS_PATH"
  --model_path "$MODEL_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --result_path "$RESULT_PATH"
  --seed "$SEED"
)

echo "[INFO] Mode: $MODE | Model: $MODEL | LLM: $LLM | Datasets: $DATASETS"
echo "[INFO] Epochs: $EPOCHS | Batch Size: $BS | Seed: $SEED"

for DS in ${DATASETS}; do
  echo ""
  echo "========================================"
  echo "[${MODE^^}] Dataset=${DS} | Model=${MODEL}"
  echo "========================================"

  # 1) Preprocessing (always needed)
  echo "[1/4] Preprocessing..."
  python main.py --task classification --job preprocessing --task_dataset "$DS" \
    "${COMMON_ARGS[@]}"

  case $MODE in
    base)
      # Base model only (no LLM)
      echo "[2/4] Training BASE model..."
      python main.py --task classification --job training --task_dataset "$DS" --test_dataset "$DS" --method base \
        --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      echo "[3/4] Testing BASE model..."
      python main.py --task classification --job testing --task_dataset "$DS" --test_dataset "$DS" --method base \
        --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      echo "[4/4] Skipped (BASE mode)"
      ;;

    pifi)
      # PiFi with specified layer (default: last layer)
      EFFECTIVE_LAYER=$LAYER
      if [[ "$EFFECTIVE_LAYER" == "-1" ]]; then
        echo "[INFO] Using last LLM layer"
      fi

      echo "[2/4] Training PiFi model with layer $EFFECTIVE_LAYER..."
      python main.py --task classification --job training --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
        --layer_num "$EFFECTIVE_LAYER" \
        --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      echo "[3/4] Testing PiFi model..."
      python main.py --task classification --job testing --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
        --layer_num "$EFFECTIVE_LAYER" \
        --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      echo "[4/4] Skipped (PIFI mode - no auto selection)"
      ;;

    pifi_ilm)
      # PiFi with ILM auto layer selection
      echo "[2/4] Training PiFi model with ILM auto-selection..."
      python main.py --task classification --job training --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
        --auto_select_layer true --selection_samples "$SEL_SAMPLES" --selection_pcs "$SEL_PCS" --selection_top_pc "$SEL_TOP_PC" \
        --selection_pooling mean --selection_dtype fp16 --selection_max_length 128 \
        --num_epochs "$EPOCHS" --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      # Read selected layer from JSON
      SEL_JSON="$RESULT_PATH/layer_selection/classification/${DS}/${MODEL}/${LLM}/selection.json"
      if [[ -f "$SEL_JSON" ]]; then
        echo "[INFO] Using selection file: $SEL_JSON"
        SELECTED_LAYER=$(python3 -c "import json; f=open('$SEL_JSON'); print(json.load(f).get('best_llm_layer', -1))")
      else
        echo "[WARN] Selection file not found; defaulting to -1"
        SELECTED_LAYER=-1
      fi

      echo "[3/4] Testing PiFi model with selected layer $SELECTED_LAYER..."
      python main.py --task classification --job testing --task_dataset "$DS" --test_dataset "$DS" --method pifi --llm "$LLM" \
        --auto_select_layer false --layer_num "$SELECTED_LAYER" \
        --batch_size "$BS" --num_workers "$WORKERS" \
        --use_wandb "$USE_WANDB" "${COMMON_ARGS[@]}"

      echo "[4/4] ILM selection complete - Layer $SELECTED_LAYER"
      ;;

    *)
      echo "[ERROR] Unknown mode: $MODE"
      echo "Valid modes: base, pifi, pifi_ilm"
      exit 1
      ;;
  esac

  echo "[DONE] Dataset=${DS} completed"
done

echo ""
echo "========================================"
echo "[${MODE^^}] All experiments completed!"
echo "========================================"
