#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# PiFi Unified Experiment Runner
#===============================================================================
# Run experiments for all tasks (classification, entailment) with unified main.py
#
# Environment Variables:
#   TASKS              - Tasks to run (default: "classification entailment")
#   DATASETS           - Datasets per task (auto-selected if not specified)
#   LLM                - LLM model (default: llama3.1)
#   MODEL              - Base model (default: bert)
#   METHOD             - Training method: base, pifi (default: pifi)
#   USE_ILM            - Enable ILM auto-selection (default: false)
#   EPOCHS             - Number of epochs (default: 3)
#   BS                 - Batch size (default: 32)
#   WORKERS            - Number of workers (default: 2)
#   SEED               - Random seed (default: 2023)
#   SEL_SAMPLES        - ILM selection samples (default: 400)
#   SEL_PCS            - ILM selection PCs (default: 16)
#   SEL_TOP_PC         - ILM selection top PCs (default: 5)
#   SKIP_PREPROCESS    - Skip preprocessing step (default: false)
#   SKIP_BASE          - Skip base method training/testing (default: false)
#   USE_WANDB          - Enable wandb logging (default: false)
#   USE_TENSORBOARD    - Enable tensorboard logging (default: false)
#
# Usage Examples:
#   # Run all tasks with default settings
#   bash scripts/run_experiments.sh
#
#   # Run classification only with ILM
#   TASKS="classification" USE_ILM=true bash scripts/run_experiments.sh
#
#   # Run entailment with specific datasets
#   TASKS="entailment" DATASETS="mnli snli" bash scripts/run_experiments.sh
#
#   # Run with custom settings
#   LLM=llama3.1 MODEL=roberta EPOCHS=5 BS=16 bash scripts/run_experiments.sh
#===============================================================================

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Get project root directory
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

log_info "PiFi root directory: $ROOT_DIR"

# Load .env if present
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
    log_info "Loaded environment from .env"
fi

# Enforce CUDA availability since CUDA wheels are required in requirements.txt
python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
if not ok:
    print("[ERROR] CUDA is not available but CUDA-only requirements are installed.")
    print("        Please install CUDA-enabled PyTorch and ensure a GPU is visible.")
    sys.exit(1)
PY

# Default configurations
TASKS=${TASKS:-"classification entailment"}
LLM=${LLM:-llama3.1}
MODEL=${MODEL:-bert}
METHOD=${METHOD:-pifi}
USE_ILM=${USE_ILM:-false}
EPOCHS=${EPOCHS:-3}
BS=${BS:-32}
WORKERS=${WORKERS:-2}
SEED=${SEED:-2023}
SEL_SAMPLES=${SEL_SAMPLES:-400}
SEL_PCS=${SEL_PCS:-16}
SEL_TOP_PC=${SEL_TOP_PC:-5}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-false}
SKIP_BASE=${SKIP_BASE:-false}
USE_WANDB=${USE_WANDB:-true}
USE_TENSORBOARD=${USE_TENSORBOARD:-true}

# If wandb enabled but no API key, default to offline to avoid failures
if [ "$USE_WANDB" == "true" ] && [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ]; then
    export WANDB_MODE=offline
    log_warn "WANDB_API_KEY not set; using WANDB_MODE=offline"
fi

# Auto-select datasets if not specified
if [ -z "${DATASETS:-}" ]; then
    if [[ "$TASKS" == *"classification"* ]] && [[ "$TASKS" == *"entailment"* ]]; then
        DATASETS="sst2 imdb mnli"
    elif [[ "$TASKS" == *"classification"* ]]; then
        DATASETS="sst2 imdb tweet_sentiment_binary tweet_offensive cola"
    elif [[ "$TASKS" == *"entailment"* ]]; then
        DATASETS="mnli snli"
    else
        DATASETS="sst2"
    fi
fi

log_info "Configuration:"
log_info "  Tasks: $TASKS"
log_info "  Datasets: $DATASETS"
log_info "  LLM: $LLM"
log_info "  Base Model: $MODEL"
log_info "  Method: $METHOD"
log_info "  Use ILM: $USE_ILM"
log_info "  Epochs: $EPOCHS"
log_info "  Batch Size: $BS"
log_info "  Workers: $WORKERS"
log_info "  Seed: $SEED"
log_info "  Use W&B: $USE_WANDB"
log_info "  Use TensorBoard: $USE_TENSORBOARD"

# Common arguments for all experiments
COMMON_ARGS=(
    --model_type "$MODEL"
    --seed "$SEED"
    --num_epochs "$EPOCHS"
    --batch_size "$BS"
    --num_workers "$WORKERS"
    --use_wandb "$USE_WANDB"
    --use_tensorboard "$USE_TENSORBOARD"
    # Localized paths to align selection/model/results under repo
    --cache_path  "$ROOT_DIR/cache"
    --preprocess_path "$ROOT_DIR/preprocessed"
    --model_path  "$ROOT_DIR/models"
    --checkpoint_path "$ROOT_DIR/checkpoints"
    --result_path "$ROOT_DIR/results"
    --log_path "$ROOT_DIR/tensorboard_logs"
)

# If WANDB_PROJECT is set, override the project name for wandb
if [ -n "${WANDB_PROJECT:-}" ]; then
    COMMON_ARGS+=( --proj_name "$WANDB_PROJECT" )
fi

# Dataset to task mapping
get_task_for_dataset() {
    local dataset=$1
    case $dataset in
        mnli|snli)
            echo "entailment"
            ;;
        sst2|imdb|cola|tweet_sentiment_binary|tweet_offensive|trec|subj|agnews|mr|cr|proscons|dbpedia|yelp_polarity|yelp_full|yahoo_answers_title|yahoo_answers_full|nsmc|filmstarts|chinese_toxicity)
            echo "classification"
            ;;
        *)
            log_error "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
}

# Main experiment loop
for DS in ${DATASETS}; do
    TASK=$(get_task_for_dataset "$DS")

    # Check if this task should be run
    if [[ ! "$TASKS" == *"$TASK"* ]]; then
        log_warn "Skipping $DS (task: $TASK) - not in TASKS list"
        continue
    fi

    echo ""
    echo "================================================================================"
    log_step "Starting experiments for $DS (task: $TASK)"
    echo "================================================================================"

    # Step 1: Preprocessing
    if [ "$SKIP_PREPROCESS" != "true" ]; then
        log_step "1/4: Preprocessing $DS"
        python main.py --task "$TASK" --job preprocessing --task_dataset "$DS" "${COMMON_ARGS[@]}" || {
            log_error "Preprocessing failed for $DS"
            continue
        }
    else
        log_warn "Skipping preprocessing for $DS"
    fi

    # Step 2: Base method training and testing
    if [ "$SKIP_BASE" != "true" ]; then
        log_step "2/4: Training base model on $DS"
        python main.py --task "$TASK" --job training --task_dataset "$DS" --test_dataset "$DS" \
            --method base "${COMMON_ARGS[@]}" || {
            log_error "Base training failed for $DS"
            continue
        }

        log_step "2/4: Testing base model on $DS"
        python main.py --task "$TASK" --job testing --task_dataset "$DS" --test_dataset "$DS" \
            --method base "${COMMON_ARGS[@]}" || {
            log_error "Base testing failed for $DS"
            continue
        }
    else
        log_warn "Skipping base method for $DS"
    fi

    # Only run PiFi if method is pifi
    if [ "$METHOD" == "pifi" ]; then
        # Step 3: PiFi method training (with optional ILM)
        log_step "3/4: Training PiFi model on $DS (LLM: $LLM, ILM: $USE_ILM)"

        PIFI_ARGS=(
            --task "$TASK"
            --job training
            --task_dataset "$DS"
            --test_dataset "$DS"
            --method pifi
            --llm_model "$LLM"
            "${COMMON_ARGS[@]}"
        )

        if [ "$USE_ILM" == "true" ]; then
            PIFI_ARGS+=(
                --auto_select_layer true
                --selection_samples "$SEL_SAMPLES"
                --selection_pcs "$SEL_PCS"
                --selection_top_pc "$SEL_TOP_PC"
                # Align to legacy ILM selection behavior
                --selection_pooling mean
                --selection_dtype fp16
                --selection_max_length 128
                --log_selection "${LOG_SELECTION:-true}"
            )
        fi

        python main.py "${PIFI_ARGS[@]}" || {
            log_error "PiFi training failed for $DS"
            continue
        }

        # Step 4: PiFi method testing
        log_step "4/4: Testing PiFi model on $DS"

        # If ILM was used, read the selected layer
        LAYER_NUM=-1
        if [ "$USE_ILM" == "true" ]; then
            SEL_JSON="$ROOT_DIR/results/layer_selection/$TASK/$DS/$MODEL/$LLM/selection.json"
            if [ -f "$SEL_JSON" ]; then
                log_info "Reading selected layer from $SEL_JSON"
                LAYER_NUM=$(python - <<PY
import json
try:
    with open('$SEL_JSON') as f:
        obj = json.load(f)
    print(int(obj.get('best_llm_layer', -1)))
except:
    print(-1)
PY
)
                log_info "Selected layer: $LAYER_NUM"
            else
                log_warn "Selection file not found: $SEL_JSON"
            fi
        fi

        python main.py --task "$TASK" --job testing --task_dataset "$DS" --test_dataset "$DS" \
            --method pifi --llm_model "$LLM" --layer_num "$LAYER_NUM" \
            --auto_select_layer false "${COMMON_ARGS[@]}" || {
            log_error "PiFi testing failed for $DS"
            continue
        }
    fi

    log_info "Completed experiments for $DS"
done

echo ""
echo "================================================================================"
log_info "All experiments completed successfully!"
echo "================================================================================"
