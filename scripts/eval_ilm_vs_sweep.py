import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from argparse import Namespace

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from selection.ilm_direct import auto_select_layer


def _load_sweep_results(root: str) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, str]]]:
    """
    Load all existing sweep test_results.json files.

    Returns:
        dict keyed by (task, dataset, model_type, llm_model) with list of (layer, acc, path).
    """
    pattern = os.path.join(
        root,
        "logs",
        "*",
        "*",
        "*",
        "pifi",
        "*",
        "*",
        "test_results.json",
    )
    files = glob.glob(pattern)
    by_key: Dict[Tuple[str, str, str, str], List[Tuple[int, float, str]]] = defaultdict(list)

    for path in files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        task = data.get("task", "")
        dataset = data.get("dataset", "")
        model_type = data.get("model_type", "")
        llm_model = data.get("llm_model", "")
        layer = int(data.get("layer_num", -1))
        acc = float(data.get("metrics", {}).get("accuracy", float("nan")))
        if not task or not dataset or not model_type or not llm_model:
            continue
        if np.isnan(acc):
            continue
        key = (task, dataset, model_type, llm_model)
        by_key[key].append((layer, acc, path))

    return by_key


def _build_ilm_args(
    task: str,
    dataset: str,
    model_type: str,
    llm_model: str,
    seed: int,
    logs_root: str,
) -> Namespace:
    """
    Build a minimal argparse.Namespace compatible with auto_select_layer().
    """
    return Namespace(
        task=task,
        task_dataset=dataset,
        test_dataset=dataset,
        model_type=model_type,
        method="pifi",
        llm_model=llm_model,
        llm=llm_model,
        selection_samples=400,
        selection_sample_ratio=0.0,
        selection_min_samples=0,
        selection_max_samples=0,
        selection_pooling="mean",
        selection_split="validation",
        selection_max_length=128,
        max_seq_len=128,
        selection_dtype="fp16",
        selection_stratified=True,
        # logging / paths
        use_wandb=False,
        log_selection=True,
        result_path=logs_root,
        cache_path="cache",
        preprocess_path="preprocessed",
        model_path="models",
        checkpoint_path="checkpoints",
        seed=seed,
        layer_num=-1,
    )


def evaluate_ilm_vs_sweep(logs_root: str, seed: int = 2023) -> None:
    """
    For each (task, dataset, model, llm) combo with sweep logs, run ILM selection
    and compare the selected layer against the sweep accuracies.
    """
    by_key = _load_sweep_results(logs_root)
    if not by_key:
        print(f"[eval] No sweep results found under {logs_root}")
        return

    summary_rows = []

    for key, sweep in sorted(by_key.items()):
        task, dataset, model_type, llm_model = key
        print("")
        print(f"===== {task} / {dataset} / {model_type} / {llm_model} =====")

        # Sort sweep by accuracy (desc)
        sweep_sorted = sorted(sweep, key=lambda x: x[1], reverse=True)
        best_layer, best_acc, _ = sweep_sorted[0]
        last_layer, last_acc, _ = max(sweep_sorted, key=lambda x: x[0])

        print(f"[sweep] total layers: {len(sweep_sorted)}")
        print(f"[sweep] best layer: {best_layer} (acc={best_acc:.4f})")
        print(f"[sweep] last layer: {last_layer} (acc={last_acc:.4f})")

        # Run ILM selection with current configuration
        args = _build_ilm_args(task, dataset, model_type, llm_model, seed, logs_root)
        try:
            ilm_layer = auto_select_layer(args)
        except Exception as e:
            print(f"[eval] ILM selection failed for {key}: {e}")
            continue

        ilm_entry = next((x for x in sweep_sorted if x[0] == ilm_layer), None)
        if ilm_entry is None:
            print(f"[eval] ILM selected layer {ilm_layer}, but no sweep result exists for this layer.")
            continue

        ilm_acc = ilm_entry[1]
        rank = sweep_sorted.index(ilm_entry) + 1
        top3 = rank <= 3
        top5 = rank <= 5

        print(f"[ilm] selected layer: {ilm_layer} (acc={ilm_acc:.4f})")
        print(f"[ilm] rank: {rank}/{len(sweep_sorted)} (top3={top3}, top5={top5})")

        summary_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "model_type": model_type,
                "llm_model": llm_model,
                "num_layers": len(sweep_sorted),
                "ilm_layer": ilm_layer,
                "ilm_acc": ilm_acc,
                "ilm_rank": rank,
                "best_layer": best_layer,
                "best_acc": best_acc,
                "last_layer": last_layer,
                "last_acc": last_acc,
                "ilm_minus_best": ilm_acc - best_acc,
                "ilm_minus_last": ilm_acc - last_acc,
                "top3": top3,
                "top5": top5,
            }
        )

    # Aggregate statistics
    if not summary_rows:
        print("[eval] No successful ILM evaluations.")
        return

    total = len(summary_rows)
    top3_count = sum(1 for r in summary_rows if r["top3"])
    top5_count = sum(1 for r in summary_rows if r["top5"])
    avg_gap_best = float(np.mean([r["ilm_minus_best"] for r in summary_rows]))
    avg_gap_last = float(np.mean([r["ilm_minus_last"] for r in summary_rows]))

    print("")
    print("===== Overall ILM vs Sweep Summary =====")
    print(f"Total combos evaluated: {total}")
    print(f"ILM in top-3: {top3_count}/{total} ({top3_count/total*100:.1f}%)")
    print(f"ILM in top-5: {top5_count}/{total} ({top5_count/total*100:.1f}%)")
    print(f"Avg (ILM - best) accuracy gap: {avg_gap_best:.4f}")
    print(f"Avg (ILM - last) accuracy gap: {avg_gap_last:.4f}")

    # Save summary to results for later inspection
    out_dir = os.path.join(logs_root, "ilm_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ilm_vs_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[eval] Saved detailed summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ILM selection vs existing sweep results.")
    parser.add_argument(
        "--logs_root",
        type=str,
        default="results",
        help="Root directory where logs/ and layer_selection/ live (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed to use for ILM selection (must match training if comparing fairly).",
    )
    args = parser.parse_args()

    evaluate_ilm_vs_sweep(args.logs_root, seed=args.seed)


if __name__ == "__main__":
    main()
