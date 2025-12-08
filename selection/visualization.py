from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from .scoring import pca_scores


def select_plot_layers(n_layers: int, best: int, spec: str, max_layers: int) -> List[int]:
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
    uniq: List[int] = []
    for v in idxs:
        if v not in uniq:
            uniq.append(v)
    return uniq[: max(1, max_layers)]


def create_selection_figures(
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

    figures: Dict[str, "plt.Figure"] = {}

    fig_line, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(list(range(len(effects))), effects, marker="o")
    if best_layer >= 0:
        ax.axvline(best_layer, color="r", linestyle="--", label=f"best={best_layer}")
        ax.legend(loc="upper right")
    ax.set_title("ILM Effect per Layer (line)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effect")
    fig_line.tight_layout()
    figures["effect_line"] = fig_line

    if sns is not None:
        fig_hm, ax = plt.subplots(figsize=(10, 2.0))
        sns.heatmap(
            [effects],
            cmap="viridis",
            cbar=True,
            xticklabels=list(range(len(effects))),
            yticklabels=["effect"],
            ax=ax,
        )
        if best_layer >= 0:
            ax.axvline(best_layer + 0.5, color="r", linestyle="--")
        ax.set_title("ILM Effect per Layer (heatmap)")
        fig_hm.tight_layout()
        figures["effect_heatmap"] = fig_hm

    if corr_mat is not None and sns is not None:
        fig_corr, ax = plt.subplots(figsize=(max(6, n_pcs / 2), max(3, corr_mat.shape[0] / 4)))
        sns.heatmap(
            corr_mat,
            cmap="magma",
            cbar=True,
            xticklabels=list(range(n_pcs)),
            yticklabels=list(range(corr_mat.shape[0])),
            ax=ax,
        )
        ax.set_title("ILM Layer x PC Label-Correlation")
        ax.set_xlabel("PC")
        ax.set_ylabel("Layer")
        fig_corr.tight_layout()
        figures["corr_heatmap"] = fig_corr

    if hidden_per_layer is not None and labels is not None:
        sel_layers = select_plot_layers(
            len(hidden_per_layer), best_layer, plot_layers_spec, max_layers
        )
        for li in sel_layers:
            X = hidden_per_layer[li]
            pcs = min(2, X.shape[0], X.shape[1])
            if pcs < 2:
                continue
            S2, _ = pca_scores(X, n_components=2)
            fig_sc, ax = plt.subplots(figsize=(4, 3))
            labs = np.array(labels)
            uniq = np.unique(labs)
            for c in uniq:
                idx = labs == c
                ax.scatter(S2[idx, 0], S2[idx, 1], s=8, alpha=0.7, label=str(int(c)))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA Scatter - Layer {li}")
            ax.legend(markerscale=2, fontsize=7, loc="best", frameon=False)
            fig_sc.tight_layout()
            figures[f"pca_scatter_layer_{li}"] = fig_sc

    return figures

