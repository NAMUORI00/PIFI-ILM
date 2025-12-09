#!/usr/bin/env python3
"""
Selection 전용 스크립트: Training 없이 Layer Selection만 실행하여 JSON 로그 생성

Usage:
    python run_selection_only.py --task classification --dataset sst2 --llm qwen2_1.5b
    python run_selection_only.py --task entailment --dataset snli --llm qwen2_7b --no_skip
"""
import os
import sys
import argparse
from types import SimpleNamespace

# 프로젝트 루트를 path에 추가 (selection/scripts/ → selection/ → PiFi/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTION_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SELECTION_DIR)
sys.path.insert(0, PROJECT_ROOT)

# selection 모듈 임포트
from selection import auto_select_layer


def build_args(
    task: str,
    dataset: str,
    llm_model: str,
    seed: int = 2023,
    device: str = "cuda",
    log_dir: str = "results/selection_logs",
    # Selection hyperparameters (기본값 사용)
    selection_samples: int = 200,
    selection_pcs: int = 16,
    selection_top_pc: int = 5,
    selection_k_shot: int = 1,
    selection_keyword_top_k: int = 12,
    selection_keyword_source: str = "tfidf",
    selection_keyword_weight: float = 0.65,
    selection_lambda_scale: float = 3.0,
    selection_patch_eval_samples: int = 0,
    selection_patch_batch_size: int = 8,
    selection_max_length: int = 128,
    selection_dtype: str = "fp16",
    selection_max_layers: int = 0,
    selection_max_heads: int = 0,
    selection_multi_layer_span: int = 0,
    max_seq_len: int = 100,
) -> SimpleNamespace:
    """
    auto_select_layer()가 필요로 하는 args 객체 생성
    SimpleNamespace를 사용하여 argparse.Namespace와 동일하게 동작
    """
    return SimpleNamespace(
        task=task,
        task_dataset=dataset,
        llm_model=llm_model,
        seed=seed,
        device=device,
        selection_log_dir=log_dir,
        selection_samples=selection_samples,
        selection_pcs=selection_pcs,
        selection_top_pc=selection_top_pc,
        selection_k_shot=selection_k_shot,
        selection_keyword_top_k=selection_keyword_top_k,
        selection_keyword_source=selection_keyword_source,
        selection_keyword_weight=selection_keyword_weight,
        selection_lambda_scale=selection_lambda_scale,
        selection_patch_eval_samples=selection_patch_eval_samples,
        selection_patch_batch_size=selection_patch_batch_size,
        selection_max_length=selection_max_length,
        selection_dtype=selection_dtype,
        selection_max_layers=selection_max_layers,
        selection_max_heads=selection_max_heads,
        selection_multi_layer_span=selection_multi_layer_span,
        max_seq_len=max_seq_len,
    )


def get_expected_log_path(log_dir: str, task: str, dataset: str, llm_model: str, seed: int) -> str:
    """JSON 로그 파일의 예상 경로 반환"""
    return os.path.join(log_dir, f"{task}_{dataset}_{llm_model}_seed{seed}.json")


def run_selection(
    task: str,
    dataset: str,
    llm_model: str,
    seed: int = 2023,
    device: str = "cuda",
    log_dir: str = "results/selection_logs",
    skip_existing: bool = True,
    **kwargs
) -> int:
    """
    단일 조합에 대해 selection 실행

    Returns:
        선택된 레이어 인덱스, 또는 에러 시 -1, 스킵 시 -2
    """
    log_path = get_expected_log_path(log_dir, task, dataset, llm_model, seed)

    # 이미 존재하는 로그 스킵
    if skip_existing and os.path.exists(log_path):
        print(f"[SKIP] Log already exists: {log_path}")
        return -2  # 스킵됨을 나타내는 특수 값

    print(f"\n{'='*60}")
    print(f"Running selection: task={task}, dataset={dataset}, llm={llm_model}, seed={seed}")
    print(f"{'='*60}")

    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)

    args = build_args(
        task=task,
        dataset=dataset,
        llm_model=llm_model,
        seed=seed,
        device=device,
        log_dir=log_dir,
        **kwargs
    )

    try:
        selected_layer = auto_select_layer(args)
        print(f"[SUCCESS] Selected layer: {selected_layer}")
        print(f"[SUCCESS] Log saved to: {log_path}")
        return selected_layer
    except Exception as e:
        print(f"[ERROR] Selection failed: {e}")
        import traceback
        traceback.print_exc()
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Run layer selection only (no training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_selection_only.py --task classification --dataset sst2 --llm qwen2_1.5b
  python run_selection_only.py --task entailment --dataset snli --llm qwen2_7b --no_skip
        """
    )

    # 필수 인자
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "entailment"],
                        help="Task type")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (sst2, cola, imdb, tweet_offensive, tweet_sentiment_binary, snli, mnli)")
    parser.add_argument("--llm", type=str, required=True,
                        choices=["qwen2_0.5b", "qwen2_1.5b", "qwen2_7b"],
                        help="LLM model")

    # 선택적 인자
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--log_dir", type=str, default="results/selection_logs",
                        help="Directory to save JSON logs")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-run even if log exists")

    # Selection hyperparameters (고급 옵션)
    parser.add_argument("--selection_samples", type=int, default=200,
                        help="Number of samples for selection")
    parser.add_argument("--selection_patch_batch_size", type=int, default=8,
                        help="Batch size for patching (reduce for large models)")
    parser.add_argument("--selection_max_length", type=int, default=128,
                        help="Max sequence length for selection")
    parser.add_argument("--selection_dtype", type=str, default="fp16",
                        choices=["fp16", "fp32"],
                        help="Data type for selection forward pass")

    args = parser.parse_args()

    skip_existing = not args.no_skip

    result = run_selection(
        task=args.task,
        dataset=args.dataset,
        llm_model=args.llm,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        skip_existing=skip_existing,
        selection_samples=args.selection_samples,
        selection_patch_batch_size=args.selection_patch_batch_size,
        selection_max_length=args.selection_max_length,
        selection_dtype=args.selection_dtype,
    )

    # 성공(>=0) 또는 스킵(-2)이면 정상 종료
    sys.exit(0 if result >= 0 or result == -2 else 1)


if __name__ == "__main__":
    main()
