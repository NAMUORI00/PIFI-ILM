#!/usr/bin/env bash
set -euo pipefail

# Test-only runner for multiple classification datasets using already trained models.
# Compares ILM-selected layer vs LAST layer without retraining.
# Optionally saves per-run metrics (loss/acc/f1) to JSON/JSONL.
#
# Env overrides:
#   MODE        : ilm,last,base,all (default: ilm,last)
#   LLM         : LLM id used for PiFi paths (default: qwen2_0.5b)
#   MODEL       : SLM model_type (default: bert)
#   DATASETS    : space-separated list (default: "sst2 imdb tweet_sentiment_binary tweet_offensive cola")
#   BS          : test batch size (default: 16)
#   WORKERS     : dataloader workers (default: 0)
#   PREPROC_MISS: if true, run preprocessing when test pkl missing (default: true)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR/Classification"

MODE=${MODE:-"ilm,last"}
LLM=${LLM:-qwen2_0.5b}
MODEL=${MODEL:-bert}
DATASETS=${DATASETS:-"sst2 imdb tweet_sentiment_binary tweet_offensive cola"}
BS=${BS:-16}
WORKERS=${WORKERS:-0}
PREPROC_MISS=${PREPROC_MISS:-true}
SAVE_JSON=${SAVE_JSON:-true}
BENCH_DIR=${BENCH_DIR:-"$ROOT_DIR/Classification/results/benchmarks"}

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

ensure_preprocessed() {
  local ds="$1"
  local test_pkl="./preprocessed/classification/${ds}/${MODEL}/test_processed.pkl"
  if [[ ! -f "$test_pkl" ]]; then
    if [[ "$PREPROC_MISS" == "true" ]]; then
      echo "[test] Missing $test_pkl → running preprocessing for $ds"
      python main.py --task classification --job=preprocessing --task_dataset="$ds" "${COMMON_ARGS[@]}" --seed 2023
    else
      echo "[test] Missing $test_pkl and PREPROC_MISS=false; skipping $ds"
      return 1
    fi
  fi
  return 0
}

test_ilm() {
  local ds="$1"
  local sel_json="./results/layer_selection/classification/${ds}/${MODEL}/${LLM}/selection.json"
  if [[ ! -f "$sel_json" ]]; then
    echo "[ILM] selection.json not found for $ds: $sel_json → skipping"
    return 0
  fi
  local layer
  layer=$(python - <<PY
import json
print(int(json.load(open('${sel_json}'))['best_llm_layer']))
PY
)
  echo "[ILM] Testing ${ds} with best_llm_layer=${layer}"
  local log_file="../logs/test_${ds}_ilm.log"
  python main.py --task classification --job=testing --task_dataset="$ds" --test_dataset="$ds" \
    --method=pifi --llm "$LLM" --layer_num "$layer" \
    --batch_size "$BS" --num_workers "$WORKERS" --use_wandb false --use_tensorboard false \
    "${COMMON_ARGS[@]}" >> "$log_file" 2>&1
  if [[ "$SAVE_JSON" == "true" ]]; then
    local summary
    summary=$(grep "Done! - TEST - " -a "$log_file" | tail -n 1 || true)
    if [[ -n "$summary" ]]; then
      local loss acc f1
      loss=$(echo "$summary" | sed -n 's/.*Loss: \([0-9.][0-9.]*\).*/\1/p')
      acc=$( echo "$summary" | sed -n 's/.*Acc: \([0-9.][0-9.]*\).*/\1/p')
      f1=$(  echo "$summary" | sed -n 's/.*F1: \([0-9.][0-9.]*\).*/\1/p')
      mkdir -p "$BENCH_DIR/$ds"
      local ts
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      local json
      json=$(cat <<J
{"task":"classification","dataset":"$ds","mode":"ILM","method":"pifi","llm":"$LLM","layer":$layer,"loss":$loss,"acc":$acc,"f1":$f1,"seed":${SEED:-2023},"timestamp":"$ts"}
J
)
      echo "$json" > "$BENCH_DIR/$ds/ilm.json"
      echo "$json" >> "$BENCH_DIR/summary.jsonl"
      echo "[ILM] Saved JSON → $BENCH_DIR/$ds/ilm.json"
    else
      echo "[ILM] Summary line not found in $log_file"
    fi
  fi
}

test_last() {
  local ds="$1"
  echo "[LAST] Testing ${ds} with layer_num=-1"
  local log_file="../logs/test_${ds}_last.log"
  python main.py --task classification --job=testing --task_dataset="$ds" --test_dataset="$ds" \
    --method=pifi --llm "$LLM" --layer_num -1 \
    --batch_size "$BS" --num_workers "$WORKERS" --use_wandb false --use_tensorboard false \
    "${COMMON_ARGS[@]}" >> "$log_file" 2>&1
  if [[ "$SAVE_JSON" == "true" ]]; then
    local summary
    summary=$(grep "Done! - TEST - " -a "$log_file" | tail -n 1 || true)
    if [[ -n "$summary" ]]; then
      local loss acc f1
      loss=$(echo "$summary" | sed -n 's/.*Loss: \([0-9.][0-9.]*\).*/\1/p')
      acc=$( echo "$summary" | sed -n 's/.*Acc: \([0-9.][0-9.]*\).*/\1/p')
      f1=$(  echo "$summary" | sed -n 's/.*F1: \([0-9.][0-9.]*\).*/\1/p')
      mkdir -p "$BENCH_DIR/$ds"
      local ts
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      local json
      json=$(cat <<J
{"task":"classification","dataset":"$ds","mode":"LAST","method":"pifi","llm":"$LLM","layer":-1,"loss":$loss,"acc":$acc,"f1":$f1,"seed":${SEED:-2023},"timestamp":"$ts"}
J
)
      echo "$json" > "$BENCH_DIR/$ds/last.json"
      echo "$json" >> "$BENCH_DIR/summary.jsonl"
      echo "[LAST] Saved JSON → $BENCH_DIR/$ds/last.json"
    else
      echo "[LAST] Summary line not found in $log_file"
    fi
  fi
}

test_base() {
  local ds="$1"
  # Base도 경로 규칙상 llm/layer 디렉토리가 포함되어 저장되므로 default 값(-1, default llm)로 저장되었을 가능성이 큼.
  # 여기서는 단순히 method=base만 지정하여 테스트 실행.
  echo "[BASE] Testing ${ds}"
  local log_file="../logs/test_${ds}_base.log"
  python main.py --task classification --job=testing --task_dataset="$ds" --test_dataset="$ds" \
    --method=base \
    --batch_size "$BS" --num_workers "$WORKERS" --use_wandb false --use_tensorboard false \
    "${COMMON_ARGS[@]}" >> "$log_file" 2>&1
  if [[ "$SAVE_JSON" == "true" ]]; then
    local summary
    summary=$(grep "Done! - TEST - " -a "$log_file" | tail -n 1 || true)
    if [[ -n "$summary" ]]; then
      local loss acc f1
      loss=$(echo "$summary" | sed -n 's/.*Loss: \([0-9.][0-9.]*\).*/\1/p')
      acc=$( echo "$summary" | sed -n 's/.*Acc: \([0-9.][0-9.]*\).*/\1/p')
      f1=$(  echo "$summary" | sed -n 's/.*F1: \([0-9.][0-9.]*\).*/\1/p')
      mkdir -p "$BENCH_DIR/$ds"
      local ts
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      local json
      json=$(cat <<J
{"task":"classification","dataset":"$ds","mode":"BASE","method":"base","llm":null,"layer":null,"loss":$loss,"acc":$acc,"f1":$f1,"seed":${SEED:-2023},"timestamp":"$ts"}
J
)
      echo "$json" > "$BENCH_DIR/$ds/base.json"
      echo "$json" >> "$BENCH_DIR/summary.jsonl"
      echo "[BASE] Saved JSON → $BENCH_DIR/$ds/base.json"
    else
      echo "[BASE] Summary line not found in $log_file"
    fi
  fi
}

for DS in ${DATASETS}; do
  echo "[test] Dataset=${DS} | Model=${MODEL} | LLM=${LLM} | MODE=${MODE}"
  if ! ensure_preprocessed "$DS"; then
    continue
  fi
  case "$MODE" in
    ilm)
      test_ilm "$DS" | tee -a "../logs/test_${DS}_ilm.log" ;;
    last)
      test_last "$DS" | tee -a "../logs/test_${DS}_last.log" ;;
    base)
      test_base "$DS" | tee -a "../logs/test_${DS}_base.log" ;;
    all|ilm,last|last,ilm)
      test_ilm "$DS"  | tee -a "../logs/test_${DS}_ilm.log"
      test_last "$DS" | tee -a "../logs/test_${DS}_last.log" ;;
    *)
      echo "[test] Unknown MODE=$MODE (use ilm|last|base|all). Skipping $DS" ;;
  esac
done

echo "[test] Completed classification test sweep."
