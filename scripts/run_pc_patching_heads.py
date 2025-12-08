import argparse
import json
import os
import sys

import numpy as np

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from selection.pc_patching_heads import run_qwen2_head_pc_patching
from core.utils import get_huggingface_model_name


def main():
    parser = argparse.ArgumentParser(description="Run ILM-style PC patching for Qwen2 attention heads.")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "entailment"])
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--llm_model", type=str, default="qwen2_0.5b",
                        help="Logical LLM name (resolved via get_huggingface_model_name).")
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_pcs", type=int, default=4)
    parser.add_argument("--top_pc", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results/head_patching",
                        help="Directory to save head importance JSON.")

    args = parser.parse_args()

    llm_id = get_huggingface_model_name(args.llm_model)

    os.makedirs(args.out_dir, exist_ok=True)

    result = run_qwen2_head_pc_patching(
        llm_id=llm_id,
        task=args.task,
        dataset=args.dataset,
        n_samples=args.n_samples,
        split=args.split,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
        n_pcs=args.n_pcs,
        top_pc=args.top_pc,
    )

    head_importance: np.ndarray = result["head_importance"]
    base_acc: float = result["base_acc"]

    out_path = os.path.join(
        args.out_dir,
        f"{args.task}_{args.dataset}_{args.llm_model}_heads.json",
    )
    payload = {
        "task": args.task,
        "dataset": args.dataset,
        "llm_model": args.llm_model,
        "base_acc": base_acc,
        "head_importance": head_importance.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[pc_patching] Base probe acc: {base_acc:.4f}")
    print(f"[pc_patching] Saved head importance to {out_path}")


if __name__ == "__main__":
    main()
