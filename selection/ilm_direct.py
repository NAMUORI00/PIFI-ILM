import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig

from core.wandb_manager import PiFiLogger


def _fisher_class_separation(*args, **kwargs):
    """Deprecated alias kept for backward compatibility."""
    from selection.scoring import fisher_class_separation

    return fisher_class_separation(*args, **kwargs)


def _logit_probe_score(*args, **kwargs):
    """Deprecated alias kept for backward compatibility."""
    from selection.scoring import logit_probe_score

    return logit_probe_score(*args, **kwargs)


def _robust_probe_score(*args, **kwargs):
    """Deprecated alias kept for backward compatibility."""
    from selection.scoring import robust_probe_score

    return robust_probe_score(*args, **kwargs)


def _rerank_topk(*args, **kwargs):
    """Deprecated alias kept for backward compatibility."""
    from selection.scoring import rerank_topk

    return rerank_topk(*args, **kwargs)


def _compute_score_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    alpha: float,
    seed: int,
    n_folds: int = 5,
    n_seeds: int = 3,
    use_confidence_weight: bool = False,
) -> float:
    """Thin wrapper delegating to selection.scoring.compute_score_for_layer."""
    from selection.scoring import compute_score_for_layer as _core_compute

    beta = float(getattr(_compute_score_for_layer, "_beta", 0.3))
    gamma = float(getattr(_compute_score_for_layer, "_gamma", 0.3))
    return _core_compute(
        X,
        y,
        mode=mode,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed,
        n_folds=n_folds,
        n_seeds=n_seeds,
        use_confidence_weight=use_confidence_weight,
    )


def auto_select_layer(args) -> int:
    """Direct ILM-style selection on LLM layers.
    Returns best_llm_layer index (0-based). Saves selection.json with effects.
    """
    task = getattr(args, "task", "classification")
    dataset = getattr(args, "task_dataset", "sst2")
    slm_type = getattr(args, "model_type", "bert")  # kept for path consistency
    llm_type = getattr(args, "llm_model", None) or getattr(args, "llm", None) or "llama3.1"
    n_samples = int(getattr(args, "selection_samples", 400))
    n_pcs = int(getattr(args, "selection_pcs", 16))
    top_pc = int(getattr(args, "selection_top_pc", 5))
    seed = int(getattr(args, "seed", 2023))
    # lambdas parsing supports str (comma-separated) or list/tuple
    _lam = getattr(args, "selection_patch_lambda", (0.5, 1.0))
    if isinstance(_lam, str):
        try:
            lambdas = tuple(float(x.strip()) for x in _lam.split(",") if x.strip())
        except Exception:
            lambdas = (0.5, 1.0)
    elif isinstance(_lam, (list, tuple)):
        lambdas = tuple(float(x) for x in _lam)
    else:
        lambdas = (0.5, 1.0)

    stride = int(getattr(args, "selection_layer_stride", 1))
    split = getattr(args, "selection_split", 'validation')
    pooling = getattr(args, "selection_pooling", 'mean')
    # max_length: use selection_max_length >0 else fall back to args.max_seq_len if present else 128
    sel_max_len = int(getattr(args, "selection_max_length", 0) or getattr(args, "max_seq_len", 128) or 128)
    dtype_pref = getattr(args, "selection_dtype", 'fp32')
    stratified = str(getattr(args, "selection_stratified", True)).lower() == "true"
    score_mode = getattr(args, "selection_score_mode", "mixed")
    score_alpha = float(getattr(args, "selection_score_alpha", 0.4))
    score_beta = float(getattr(args, "selection_score_beta", 0.3))  # fisher
    score_gamma = float(getattr(args, "selection_score_gamma", 0.3))  # silhouette
    _compute_score_for_layer._beta = score_beta
    _compute_score_for_layer._gamma = score_gamma

    # Robust selection parameters (K-fold CV + multi-seed + selectivity)
    robust_selection = str(getattr(args, "robust_selection", False)).lower() in ("true", "1", "yes")
    selection_n_folds = int(getattr(args, "selection_n_folds", 5))
    selection_n_seeds = int(getattr(args, "selection_n_seeds", 3))
    # Confidence weighting is OFF by default (pure selectivity, no heuristic)
    use_confidence_weight = str(getattr(args, "selection_use_confidence_weight", False)).lower() in ("true", "1", "yes")
    if robust_selection:
        score_mode = "robust"
        conf_msg = "+ confidence weight" if use_confidence_weight else "(pure selectivity)"
        print(f"[selection] Robust mode enabled: {selection_n_folds}-fold CV x {selection_n_seeds} seeds {conf_msg}")

    # Resolve huggingface model id for LLM
    from core.utils import get_huggingface_model_name

    llm_id = get_huggingface_model_name(llm_type)

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

    depth_bias = float(getattr(args, "selection_depth_bias", 0.3))
    num_layers = len(hidden_per_layer)

    # 1st stage: coarse scoring (logit probe or robust probe)
    effects: List[float] = []
    for Li, X in enumerate(hidden_per_layer):
        if stride > 1 and (Li % stride != 0):
            effects.append(0.0)
            continue
        try:
            eff = _compute_score_for_layer(X, y, mode=score_mode, alpha=score_alpha, seed=seed,
                                           n_folds=selection_n_folds, n_seeds=selection_n_seeds,
                                           use_confidence_weight=use_confidence_weight)
        except Exception as e:
            print(f"[selection] LLM layer {Li} scoring failed: {e}; using 0")
            eff = 0.0
        # depth prior: slightly favor shallower layers
        depth_weight = 1.0 - depth_bias * (Li / max(1, num_layers - 1))
        effects.append(eff * depth_weight)

    # Top-k rerank with same score (lightweight)
    k = int(getattr(args, "selection_topk_rerank", 3))
    eff_np = np.array(effects)
    top_idx = np.argsort(-eff_np)[: max(1, k)]
    top_scores = eff_np[top_idx]
    from selection.scoring import rerank_topk

    best_llm_layer = rerank_topk(list(top_idx), list(top_scores))

    # Compute layer x PC correlation matrix for visualization (no extra forward pass)
    corr_mat = None
    try:
        corr_rows: List[np.ndarray] = []
        for X in hidden_per_layer:
            # Guard against tiny sample sizes
            from selection.scoring import pca_scores, label_correlation

            pcs = min(n_pcs, X.shape[0], X.shape[1])
            if pcs <= 0:
                corr_rows.append(np.zeros((n_pcs,), dtype=np.float32))
                continue
            S, comps = pca_scores(X, n_components=pcs)
            corrs = label_correlation(S, y)
            # Pad/trim to n_pcs for consistent heatmap width
            row = np.zeros((n_pcs,), dtype=np.float32)
            row[:pcs] = corrs[:pcs]
            corr_rows.append(row)
        corr_mat = np.stack(corr_rows, axis=0)  # (layers, n_pcs)
    except Exception as _:
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
    }

    # Create figures if logging/saving plots is enabled
    figures = None
    should_log_plots = getattr(args, 'log_selection', True) or getattr(args, 'save_selection_plots', False)
    if should_log_plots and getattr(args, 'log_selection_pca', True):
        try:
            figures = _create_selection_figures(
                effects, best_llm_layer, n_pcs,
                corr_mat=corr_mat,
                hidden_per_layer=hidden_per_layer,
                labels=y,
                plot_layers_spec=getattr(args, 'selection_plot_layers', 'best,first,mid,last'),
                max_layers=getattr(args, 'selection_plot_max_layers', 6)
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
    if figures:
        for fig in figures.values():
            plt.close(fig)

    print(f"[selection] Best LLM layer: {best_llm_layer}")
    return best_llm_layer


def _select_plot_layers(n_layers: int, best: int, spec: str, max_layers: int) -> List[int]:
    """Resolve which layers to plot based on spec string."""
    if not spec:
        spec = 'best,first,mid,last'
    if spec.strip().lower() == 'all':
        idxs = list(range(n_layers))
    else:
        idxs = []
        tokens = [t.strip().lower() for t in spec.split(',') if t.strip()]
        for t in tokens:
            if t == 'best' and 0 <= best < n_layers:
                idxs.append(best)
            elif t == 'first':
                idxs.append(0)
            elif t == 'last':
                idxs.append(n_layers-1)
            elif t == 'mid':
                idxs.append(max(0, n_layers//2))
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
    return uniq[:max(1, max_layers)]


def _create_selection_figures(effects: List[float], best_layer: int,
                               n_pcs: int,
                               corr_mat: Optional[np.ndarray] = None,
                               hidden_per_layer: Optional[List[np.ndarray]] = None,
                               labels: Optional[np.ndarray] = None,
                               plot_layers_spec: str = 'best,first,mid,last',
                               max_layers: int = 6) -> Dict[str, 'plt.Figure']:
    """
    Create all selection visualization figures.

    Args:
        effects: List of effect scores per layer
        best_layer: Selected best layer index
        n_pcs: Number of principal components used
        corr_mat: Optional layer x PC correlation matrix
        hidden_per_layer: Optional list of hidden states per layer
        labels: Optional labels for PCA scatter plots
        plot_layers_spec: Specification for which layers to plot
        max_layers: Maximum number of layers to plot

    Returns:
        Dict of figure name -> matplotlib Figure
    """
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
