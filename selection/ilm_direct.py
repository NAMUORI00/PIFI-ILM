"""ILM-style direct layer selection for LLM→SLM transfer.

Supports two scoring methods:
1. ILM-PCA: PC-label correlation (Sun et al., ACL 2025)
2. MDL: Minimum Description Length (Voita & Titov, EMNLP 2020)
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig

from core.wandb_manager import PiFiLogger

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
        or getattr(args, "max_seq_len", 0)
        or 0
    )
    dtype_pref = getattr(args, "selection_dtype", "fp32")
    stratified = str(getattr(args, "selection_stratified", True)).lower() == "true"
    score_mode = getattr(args, "selection_score_mode", "ilm_pca")
    silhouette_weight = float(getattr(args, "selection_silhouette_weight", 0.0))
    depth_bias = float(getattr(args, "selection_depth_bias", 0.0))
    fisher_weight = float(getattr(args, "selection_fisher_weight", 0.0))
    entropy_weight = float(getattr(args, "selection_entropy_weight", 0.0))
    head_alpha = float(getattr(args, "selection_head_alpha", 0.0))
    match_retry_threshold = float(getattr(args, "selection_match_retry_threshold", 0.2))
    match_fallback = str(getattr(args, "selection_match_fallback", "none"))
    # Task-aware default max_length if not provided
    if sel_max_len == 0:
        ds = (dataset or "").lower()
        if ds in {"imdb"}:
            sel_max_len = 256
        elif ds in {"mnli", "snli"}:
            sel_max_len = 192
        elif ds in {"tweet_offensive", "tweet_sentiment_binary"}:
            sel_max_len = 128
        else:
            sel_max_len = 128
        print(f"[selection] selection_max_length not set; using task-aware default {sel_max_len} for {ds}")

    # Resolve huggingface model id for LLM
    from core.utils import get_huggingface_model_name

    llm_id = get_huggingface_model_name(llm_type)

    # Head-level PC patching mode (Qwen2-specific)
    if score_mode == "ilm_head_patching":
        from selection.pc_patching_heads import run_qwen2_head_pc_patching

        sel_device = getattr(args, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        k_shot = int(getattr(args, "selection_k_shot", 1))
        print(
            f"[selection] Running head-level PC patching for {llm_type} "
            f"on {task}/{dataset} (samples={n_samples}, k_shot={k_shot})"
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
            k_shot=k_shot,
            prompt_max_length=sel_max_len,
        )
        head_importance = result["head_importance"]  # (layers, heads)
        layer_scores = head_importance.sum(axis=1) if head_importance.size > 0 else np.zeros(
            (head_importance.shape[0],), dtype=np.float32
        )

        effects = layer_scores.tolist()
        if depth_bias != 0 and len(effects) > 1:
            num_layers = len(effects)
            effects = [
                eff - depth_bias * (idx / (num_layers - 1))
                for idx, eff in enumerate(effects)
            ]
        if not effects:
            print("[selection] Head patching produced empty scores; defaulting to mid layer")
            cfg = AutoConfig.from_pretrained(llm_id)
            return max(0, cfg.num_hidden_layers // 2)

        # Blend with optional base layer scores if provided via args (not available here),
        # so we just apply alpha on head_importance-derived effects.
        if head_alpha > 0 and head_importance.size > 0:
            # Normalize head_importance per layer
            head_layer = head_importance.sum(axis=1)
            h_norm = head_layer / (np.max(head_layer) + 1e-8)
            effects = [
                (1 - head_alpha) * eff + head_alpha * h_norm[i]
                for i, eff in enumerate(effects)
            ]

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
            "head_alpha": head_alpha if head_alpha != 0 else None,
        }

        figures = None
        try:
            logger = PiFiLogger.from_args(args)
            out_path = logger.log_selection(effects, best_llm_layer, metadata, figures)
            print(f"[selection] Saved selection to {out_path}")
        except Exception as e:
            print(f"[selection] Logging failed (non-fatal): {e}")

        print(f"[selection] Best LLM layer: {best_llm_layer} (mode={score_mode})")
        return best_llm_layer

    # Collect data
    from selection.data import resolve_dataset

    texts, labels = resolve_dataset(
        task, dataset, n_samples=n_samples, seed=seed, split=split, stratified=stratified
    )
    if len(texts) < 10:
        print("[selection] Not enough samples; defaulting to mid layer")
        cfg = AutoConfig.from_pretrained(llm_id)
        return max(0, cfg.num_hidden_layers // 2)

    # Collect pooled hidden per layer for LLM
    from selection.embeddings import get_llm_hidden_layers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (dtype_pref == "fp16" and device == "cuda") else torch.float32
    use_label_tokens = score_mode == "ilm_pca_unembed"
    label_embs = None

    if use_label_tokens:
        from selection.prompts import build_label_prompts
        from selection.label_embeddings import get_label_token_hidden_per_layer

        k_shot = int(getattr(args, "selection_k_shot", 1))
        prompts, label_token_strs, prompt_labels = build_label_prompts(texts, labels, k_shot=k_shot)
        hidden_per_layer, label_embs, match_stats = get_label_token_hidden_per_layer(
            llm_id,
            prompts,
            label_token_strs,
            device=device,
            max_length=sel_max_len,
            batch_size=8,
            dtype=dtype,
            return_label_embs=True,
            fallback=match_fallback,
        )
        if len(label_token_strs) != len(prompts):
            print(f"[selection] Warning: label token alignment mismatch ({len(label_token_strs)} vs {len(prompts)})")

        # Retry with longer max_length if match fail rate is high
        total_match = max(1, match_stats["match_success"] + match_stats["match_fail"])
        fail_rate = match_stats["match_fail"] / total_match
        if fail_rate > match_retry_threshold:
            retry_len = int(sel_max_len * 1.5)
            print(
                f"[selection] High match fail rate ({fail_rate:.2f}); retrying with max_length={retry_len}"
            )
            hidden_per_layer, label_embs, match_stats = get_label_token_hidden_per_layer(
                llm_id,
                prompts,
                label_token_strs,
                device=device,
                max_length=retry_len,
                batch_size=8,
                dtype=dtype,
                return_label_embs=True,
                fallback=match_fallback,
            )
            sel_max_len = retry_len
        y_arr = np.array(prompt_labels, dtype=int)
    else:
        hidden_per_layer, _, match_stats = get_llm_hidden_layers(
            llm_id,
            texts,
            device=device,
            max_length=sel_max_len,
            batch_size=16,
            dtype=dtype,
            pooling=pooling,
            l2_norm=True,
        )
        y_arr = np.array(labels)

    num_layers = len(hidden_per_layer)
    print(f"[selection] Scoring {num_layers} layers with mode={score_mode}")

    # Score each layer
    from selection.scoring import compute_score_for_layer

    effects: List[float] = []
    for Li, X in enumerate(hidden_per_layer):
        if stride > 1 and (Li % stride != 0):
            effects.append(0.0)
            continue
        try:
            eff = compute_score_for_layer(
                X,
                y_arr,
                mode=score_mode,
                n_pcs=n_pcs,
                top_pc=top_pc,
                seed=seed,
                silhouette_weight=silhouette_weight,
                fisher_weight=fisher_weight,
                entropy_weight=entropy_weight,
                label_embs=label_embs,
            )
        except Exception as e:
            print(f"[selection] LLM layer {Li} scoring failed: {e}; using fallback")
            eff = 0.0
        # Depth bias: penalize deeper layers if requested to counter last-layer preference
        if depth_bias != 0 and num_layers > 1:
            depth_penalty = depth_bias * (Li / (num_layers - 1))
            eff -= depth_penalty
        effects.append(eff)

    # Select best layer (argmax for all modes - MDL is already negated)
    best_llm_layer = int(np.argmax(effects))

    # Compute layer x PC correlation matrix for visualization (ILM-PCA only)
    corr_mat = None
    if score_mode == "ilm_pca":
        try:
            from selection.scoring import label_correlation, pca_scores

            corr_rows: List[np.ndarray] = []
            for X in hidden_per_layer:
                pcs = min(n_pcs, X.shape[0], X.shape[1])
                if pcs <= 0:
                    corr_rows.append(np.zeros((n_pcs,), dtype=np.float32))
                    continue
                S, _ = pca_scores(X, n_components=pcs)
                corrs = label_correlation(S, y)
                row = np.zeros((n_pcs,), dtype=np.float32)
                row[:pcs] = corrs[:pcs]
                corr_rows.append(row)
            corr_mat = np.stack(corr_rows, axis=0)
        except Exception:
            corr_mat = None

    # Prepare metadata
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
        "silhouette_weight": silhouette_weight if score_mode.startswith("ilm_pca") else None,
        "depth_bias": depth_bias if depth_bias != 0 else None,
        "use_label_tokens": use_label_tokens,
        "fisher_weight": fisher_weight if fisher_weight != 0 else None,
        "entropy_weight": entropy_weight if entropy_weight != 0 else None,
        "head_alpha": head_alpha if head_alpha != 0 else None,
        "match_success": match_stats["match_success"],
        "match_fail": match_stats["match_fail"],
        "selection_max_length": sel_max_len,
    }

    # Create figures if logging/saving plots is enabled
    figures = None
    should_log_plots = getattr(args, "log_selection", True) or getattr(
        args, "save_selection_plots", False
    )
    if should_log_plots and getattr(args, "log_selection_pca", True):
        try:
            figures = _create_selection_figures(
                effects,
                best_llm_layer,
                n_pcs,
                corr_mat=corr_mat,
                hidden_per_layer=hidden_per_layer,
                labels=y,
                plot_layers_spec=getattr(args, "selection_plot_layers", "best,first,mid,last"),
                max_layers=getattr(args, "selection_plot_max_layers", 6),
            )
        except Exception as e:
            print(f"[selection] Figure creation failed (non-fatal): {e}")
            figures = {}

    # Use unified PiFiLogger for logging (W&B + local)
    try:
        logger = PiFiLogger.from_args(args)
        out_path = logger.log_selection(effects, best_llm_layer, metadata, figures)
        print(f"[selection] Saved selection to {out_path}")
    except Exception as e:
        print(f"[selection] Logging failed (non-fatal): {e}")

    # Close figures to free memory
    if figures and plt is not None:
        for fig in figures.values():
            plt.close(fig)

    print(f"[selection] Best LLM layer: {best_llm_layer} (mode={score_mode})")
    return best_llm_layer


def _select_plot_layers(n_layers: int, best: int, spec: str, max_layers: int) -> List[int]:
    """Resolve which layers to plot based on spec string."""
    if not spec:
        spec = "best,first,mid,last"
    if spec.strip().lower() == "all":
        idxs = list(range(n_layers))
    else:
        idxs = []
        tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
        for t in tokens:
            if t == "best" and 0 <= best < n_layers:
                idxs.append(best)
            elif t == "first":
                idxs.append(0)
            elif t == "last":
                idxs.append(n_layers - 1)
            elif t == "mid":
                idxs.append(max(0, n_layers // 2))
            else:
                try:
                    v = int(t)
                    if 0 <= v < n_layers:
                        idxs.append(v)
                except Exception:
                    pass
    # unique and cap
    uniq = []
    for v in idxs:
        if v not in uniq:
            uniq.append(v)
    return uniq[: max(1, max_layers)]


def _create_selection_figures(
    effects: List[float],
    best_layer: int,
    n_pcs: int,
    corr_mat: Optional[np.ndarray] = None,
    hidden_per_layer: Optional[List[np.ndarray]] = None,
    labels: Optional[np.ndarray] = None,
    plot_layers_spec: str = "best,first,mid,last",
    max_layers: int = 6,
) -> Dict[str, "plt.Figure"]:
    """Create all selection visualization figures."""
    if plt is None:
        return {}

    from selection.visualization import create_selection_figures as _viz_create

    return _viz_create(
        effects,
        best_layer,
        n_pcs,
        corr_mat=corr_mat,
        hidden_per_layer=hidden_per_layer,
        labels=labels,
        plot_layers_spec=plot_layers_spec,
        max_layers=max_layers,
    )
