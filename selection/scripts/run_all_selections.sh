#!/bin/bash
#
# 모든 태스크/데이터셋/LLM 조합에 대해 Selection Log 생성
#
# Usage:
#   ./run_all_selections.sh           # 전체 실행 (존재하는 로그 스킵)
#   ./run_all_selections.sh --force   # 강제 재실행

# 프로젝트 루트 디렉토리 (selection/scripts/ → selection/ → PiFi/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELECTION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SELECTION_DIR")"
cd "$PROJECT_ROOT"

# 결과 저장 디렉토리
LOG_DIR="results/selection_logs"
mkdir -p "$LOG_DIR"

# Python 실행 (가상환경 또는 시스템 Python)
PYTHON="python"

# 옵션 처리
SKIP_FLAG=""
if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
    SKIP_FLAG="--no_skip"
    echo "[INFO] Force mode: will re-run all selections"
fi

# =============================================================================
# 실행 대상 정의
# =============================================================================

# Classification 데이터셋
CLASSIFICATION_DATASETS=("sst2" "cola" "imdb" "tweet_offensive" "tweet_sentiment_binary")

# TextualEntailment 데이터셋
ENTAILMENT_DATASETS=("snli" "mnli")

# LLM 모델 (0.5b 제외 - 결과가 거의 없음)
LLMS=("qwen2_1.5b" "qwen2_7b")

# Seed
SEED=2023

# GPU 설정
DEVICE="cuda"

# =============================================================================
# 실행 함수
# =============================================================================

run_selection() {
    local task=$1
    local dataset=$2
    local llm=$3

    echo ""
    echo "========================================"
    echo "Task: $task | Dataset: $dataset | LLM: $llm"
    echo "========================================"

    # GPU 메모리 정리를 위해 잠시 대기
    sleep 2

    # 7B 모델은 배치 크기 축소
    local batch_size=8
    if [ "$llm" = "qwen2_7b" ]; then
        batch_size=4
    fi

    "$PYTHON" selection/scripts/run_selection_only.py \
        --task "$task" \
        --dataset "$dataset" \
        --llm "$llm" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --log_dir "$LOG_DIR" \
        --selection_patch_batch_size "$batch_size" \
        $SKIP_FLAG

    local status=$?
    if [ $status -ne 0 ]; then
        echo "[WARNING] Selection failed for $task/$dataset/$llm"
    fi

    return 0  # 개별 실패해도 전체 스크립트는 계속 실행
}

# =============================================================================
# 메인 실행
# =============================================================================

echo "============================================"
echo "PiFi Selection Log Generator"
echo "============================================"
echo "Log directory: $LOG_DIR"
echo "Python: $PYTHON"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo ""

START_TIME=$(date +%s)

# Classification 태스크 실행
echo ""
echo ">>> Running Classification tasks..."
for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
    for llm in "${LLMS[@]}"; do
        run_selection "classification" "$dataset" "$llm"
    done
done

# TextualEntailment 태스크 실행
echo ""
echo ">>> Running TextualEntailment tasks..."
for dataset in "${ENTAILMENT_DATASETS[@]}"; do
    for llm in "${LLMS[@]}"; do
        run_selection "entailment" "$dataset" "$llm"
    done
done

# 완료 메시지
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "============================================"
echo "All selections completed!"
echo "Time elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "============================================"

# 생성된 로그 파일 목록 출력
echo ""
echo "Generated log files:"
ls -la "$LOG_DIR"/*.json 2>/dev/null || echo "(no logs generated)"
