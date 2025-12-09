"""ILM-style direct layer selection for LLM→SLM transfer (no visualization).

Supports two scoring methods:
1. ILM-PCA: PC-label correlation
2. MDL: Minimum Description Length
3. ILM head-level PC patching (Qwen2-only)
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig

try:
    # When called from Classification/TextualEntailment scripts, sys.path already includes their utils.
    from utils.utils import get_huggingface_model_name
except Exception:
    get_huggingface_model_name = None  # Will fallback later if needed


def _resolve_llm_id(llm_type: str) -> str:
    if get_huggingface_model_name:
        return get_huggingface_model_name(llm_type)
    # Minimal fallback mapping (kept in sync with Classification/TextualEntailment utils)
    name = (llm_type or "").lower()
    mapping = {
        "llama2": "meta-llama/Llama-2-7b-hf",
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "llama3.1": "meta-llama/Meta-Llama-3.1-8B",
        "llama3.1_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral0.1": "mistralai/Mistral-7B-v0.1",
        "mistral0.3": "mistralai/Mistral-7B-v0.3",
        "qwen2_7b": "Qwen/Qwen2-7B",
        "qwen2_0.5b": "Qwen/Qwen2-0.5B",
        "qwen2_1.5b": "Qwen/Qwen2-1.5B",
        "qwen2_72b": "Qwen/Qwen2-72B",
        "gemma2": "google/gemma-2-9b",
        "falcon": "tiiuae/falcon-7b",
        "kollama": "beomi/Llama-3-Open-Ko-8B",
        "gerllama": "DiscoResearch/Llama3-German-8B",
        "chillama": "hfl/llama-3-chinese-8b",
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f"Unknown llm_model '{llm_type}'")


def _save_selection_json(
    effects: List[float],
    best_layer: int,
    args,
    metadata: Dict[str, object],
) -> Optional[str]:
    """Save selection results to a consistent path; ignore failures."""
    try:
        result_path = getattr(args, "result_path", "results")
        task = getattr(args, "task", "classification")
        dataset = getattr(args, "task_dataset", "sst2")
        model_type = getattr(args, "model_type", "bert")
        llm_type = getattr(args, "llm_model", "")

        out_dir = os.path.join(result_path, "layer_selection", task, dataset, model_type)
        if getattr(args, "method", "") == "pifi" and llm_type:
            out_dir = os.path.join(out_dir, llm_type)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "selection.json")

        payload = {
            **metadata,
            "effects": effects,
            "best_llm_layer": int(best_layer),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[selection] Saved selection to {out_path}")
        return out_path
    except Exception as e:
        print(f"[selection] Failed to save selection.json: {e}")
        return None


def auto_select_layer(args) -> int:
    """Direct ILM-style selection on LLM layers.

    Returns best_llm_layer index (0-based). Saves selection.json with effects.
    """
    task = getattr(args, "task", "classification")
    dataset = getattr(args, "task_dataset", "sst2")
    slm_type = getattr(args, "model_type", "bert")
    llm_type = getattr(args, "llm_model", None) or getattr(args, "llm", None) or "llama3.1"
    n_samples = int(getattr(args, "selection_samples", 400))
    n_pcs = int(getattr(args, "selection_pcs", 16))
    top_pc = int(getattr(args, "selection_top_pc", 5))
    seed = int(getattr(args, "seed", 2023))
    stride = int(getattr(args, "selection_layer_stride", 1))
    split = getattr(args, "selection_split", "validation")
    pooling = getattr(args, "selection_pooling", "mean")
    sel_max_len = int(
        getattr(args, "selection_max_length", 0)
        or getattr(args, "max_seq_len", 128)
        or 128
    )
    dtype_pref = getattr(args, "selection_dtype", "fp32")
    stratified = str(getattr(args, "selection_stratified", True)).lower() == "true"
    score_mode = getattr(args, "selection_score_mode", "ilm_pca")
    mdl_n_portions = int(getattr(args, "mdl_n_portions", 10))

    llm_id = _resolve_llm_id(llm_type)

    if score_mode == "ilm_head_patching":
        from selection.pc_patching_heads import run_qwen2_head_pc_patching

        sel_device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"[selection] Running head-level PC patching for {llm_type} "
            f"on {task}/{dataset} (samples={n_samples})"
        )
        result = run_qwen2_head_pc_patching(
            llm_id=llm_id,
            task=task,
            dataset=dataset,
            n_samples=n_samples,
            split=split,
            max_length=sel_max_len,
            device=sel_device,
            batch_size=16,
            n_pcs=n_pcs,
            top_pc=top_pc,
            seed=seed,
        )
        head_importance = result["head_importance"]  # (layers, heads)
        if head_importance.size == 0:
            print("[selection] Head patching produced empty scores; defaulting to mid layer")
            cfg = AutoConfig.from_pretrained(llm_id)
            return max(0, cfg.num_hidden_layers // 2)

        layer_scores = head_importance.sum(axis=1)
        effects = layer_scores.tolist()
        best_llm_layer = int(np.argmax(effects))

        metadata = {
            "task": task,
            "dataset": dataset,
            "slm_type": slm_type,
            "llm_type": llm_type,
            "n_samples": n_samples,
            "n_pcs": n_pcs,
            "top_pc": top_pc,
            "seed": seed,
            "score_mode": score_mode,
            "base_probe_acc": float(result.get("base_acc", 0.0)),
        }
        _save_selection_json(effects, best_llm_layer, args, metadata)
        print(f"[selection] Best LLM layer: {best_llm_layer} (mode={score_mode})")
        return best_llm_layer

    from selection.data import resolve_dataset

    texts, labels = resolve_dataset(
        task, dataset, n_samples=n_samples, seed=seed, split=split, stratified=stratified
    )
    if len(texts) < 10:
        print("[selection] Not enough samples; defaulting to mid layer")
        cfg = AutoConfig.from_pretrained(llm_id)
        return max(0, cfg.num_hidden_layers // 2)

    from selection.embeddings import get_llm_hidden_layers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (dtype_pref == "fp16" and device == "cuda") else torch.float32
    hidden_per_layer = get_llm_hidden_layers(
        llm_id,
        texts,
        device=device,
        max_length=sel_max_len,
        batch_size=16,
        dtype=dtype,
        pooling=pooling,
        l2_norm=True,
    )
    y = np.array(labels)

    num_layers = len(hidden_per_layer)
    print(f"[selection] Scoring {num_layers} layers with mode={score_mode}")

    from selection.scoring import compute_score_for_layer

    effects: List[float] = []
    for Li, X in enumerate(hidden_per_layer):
        if stride > 1 and (Li % stride != 0):
            effects.append(float("-inf") if score_mode == "mdl" else 0.0)
            continue
        try:
            eff = compute_score_for_layer(
                X,
                y,
                mode=score_mode,
                n_pcs=n_pcs,
                top_pc=top_pc,
                mdl_n_portions=mdl_n_portions,
                seed=seed,
            )
        except Exception as e:
            print(f"[selection] LLM layer {Li} scoring failed: {e}; using fallback")
            eff = float("-inf") if score_mode == "mdl" else 0.0
        effects.append(eff)

    best_llm_layer = int(np.argmax(effects))

    metadata = {
        "task": task,
        "dataset": dataset,
        "slm_type": slm_type,
        "llm_type": llm_type,
        "n_samples": n_samples,
        "n_pcs": n_pcs,
        "top_pc": top_pc,
        "seed": seed,
        "score_mode": score_mode,
        "stride": stride,
        "pooling": pooling,
        "selection_max_length": sel_max_len,
    }

    _save_selection_json(effects, best_llm_layer, args, metadata)

    print(f"[selection] Best LLM layer: {best_llm_layer} (mode={score_mode})")
    return best_llm_layer
