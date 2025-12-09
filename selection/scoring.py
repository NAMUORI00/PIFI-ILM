"""Layer selection scoring functions (ILM-focused)."""
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress sklearn convergence warnings in scoring utilities
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply PCA and return scores and components."""
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    S = pca.fit_transform(X)
    return S, pca.components_


def label_correlation(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute absolute correlation between PC scores and labels."""
    y = np.asarray(y)
    if len(np.unique(y)) <= 2:
        yv = y.astype(float)
        corrs = []
        for j in range(scores.shape[1]):
            s = scores[:, j]
            if s.std() < 1e-8 or yv.std() < 1e-8:
                corrs.append(0.0)
            else:
                corrs.append(abs(np.corrcoef(s, yv)[0, 1]))
        return np.array(corrs)

    # Multi-class: max correlation across one-vs-rest
    K = int(np.max(y)) + 1
    corrs = []
    for j in range(scores.shape[1]):
        s = scores[:, j]
        vals = []
        for k in range(K):
            yk = (y == k).astype(float)
            if s.std() < 1e-8 or yk.std() < 1e-8:
                vals.append(0.0)
            else:
                vals.append(abs(np.corrcoef(s, yk)[0, 1]))
        corrs.append(max(vals))
    return np.array(corrs)


def ilm_pca_score(
    X: np.ndarray, y: np.ndarray, n_pcs: int = 16, top_pc: int = 5
) -> float:
    """
    ILM-inspired PCA score: average label-correlation of top PCs.

    For a given layer representation X, we:
      1) run PCA to get PC scores per sample,
      2) measure correlation between each PC score and labels,
      3) return the mean correlation of the top-k PCs.

    This approximates how strongly input-label mappings are encoded
    in a small subspace of the layer representation.

    References:
        Sun et al. (ACL 2025) - Interpret and Improve ICL via Input-Label Mappings
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2 or X.shape[0] < 10 or X.shape[1] == 0:
        return 0.0

    pcs = min(n_pcs, X.shape[0], X.shape[1])
    if pcs <= 0:
        return 0.0

    try:
        S, _ = pca_scores(X, n_components=pcs)
    except Exception:
        return 0.0

    corrs = label_correlation(S, y)
    if corrs.size == 0:
        return 0.0

    k = min(top_pc, corrs.size)
    top_corrs = np.sort(corrs)[-k:]
    return float(np.mean(top_corrs))


def silhouette_signal(X: np.ndarray, y: np.ndarray) -> float:
    """Silhouette score as a weak cluster-compactness signal."""
    y = np.asarray(y)
    if X.ndim != 2 or len(np.unique(y)) < 2 or X.shape[0] < 10:
        return 0.0
    try:
        return float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        return 0.0


def fisher_signal(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher score (between-class scatter / within-class scatter)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)
    if X.ndim != 2 or X.shape[0] < 10 or len(classes) < 2:
        return 0.0
    overall_mean = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        idx = y == c
        if not np.any(idx):
            continue
        Xc = X[idx]
        n_c = Xc.shape[0]
        mean_c = Xc.mean(axis=0)
        between += n_c * np.sum((mean_c - overall_mean) ** 2)
        within += n_c * np.sum((Xc - mean_c) ** 2)
    denom = within + 1e-8
    return float(between / denom)


def predictive_entropy_signal(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    train_ratio: float = 0.8,
) -> float:
    """
    Predictive entropy (lower = better) using a logistic probe on X.
    Returns negative mean entropy so higher is better for scoring.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] < 20 or len(np.unique(y)) < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = max(1, int(train_ratio * len(y)))
    train_idx, val_idx = idx[:cut], idx[cut:]
    if len(val_idx) < 5:
        return 0.0
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    try:
        clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()
        return -float(entropy)  # higher is better
    except Exception:
        return 0.0


def ilm_pca_unembed_score(
    X: np.ndarray,
    label_embs: np.ndarray,
    n_pcs: int = 16,
    top_pc: int = 5,
) -> float:
    """
    ILM-PCA with unembedding: cosine similarity of PCs to label embeddings.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 10 or X.shape[1] == 0:
        return 0.0
    if label_embs is None or label_embs.size == 0:
        return 0.0

    pcs = min(n_pcs, X.shape[0], X.shape[1])
    if pcs <= 0:
        return 0.0
    try:
        _, comps = pca_scores(X, n_components=pcs)
    except Exception:
        return 0.0

    lab = np.asarray(label_embs, dtype=np.float32)
    if lab.ndim == 1:
        lab = lab[None, :]
    lab_mean = lab.mean(axis=0)
    lab_norm = lab_mean / max(np.linalg.norm(lab_mean), 1e-8)
    scores = []
    for j in range(comps.shape[0]):
        pc = comps[j]
        pc_norm = pc / max(np.linalg.norm(pc), 1e-8)
        scores.append(abs(float(np.dot(pc_norm, lab_norm))))
    if not scores:
        return 0.0
    k = min(top_pc, len(scores))
    return float(np.mean(sorted(scores)[-k:]))


def compute_score_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "ilm_pca",
    n_pcs: int = 16,
    top_pc: int = 5,
    seed: int = 0,
    silhouette_weight: float = 0.0,
    fisher_weight: float = 0.0,
    entropy_weight: float = 0.0,
    label_embs: np.ndarray | None = None,
) -> float:
    """
    Compute layer selection score.

    Args:
        X: Hidden representations (n_samples, hidden_dim)
        y: Labels (n_samples,)
        mode: "ilm_pca", "ilm_pca_silhouette", or "ilm_pca_unembed"
        n_pcs: Number of PCs for ILM-PCA
        top_pc: Top-k PCs to average for ILM-PCA
        seed: Random seed

    Returns:
        Score (higher is better for all modes)
    """
    mode = (mode or "ilm_pca").lower()

    if mode == "ilm_pca":
        base = ilm_pca_score(X, y, n_pcs=n_pcs, top_pc=top_pc)
        if silhouette_weight > 0:
            sil = silhouette_signal(X, y)
            return float(base + silhouette_weight * sil)
        return base

    if mode == "ilm_pca_silhouette":
        base = ilm_pca_score(X, y, n_pcs=n_pcs, top_pc=top_pc)
        sil = silhouette_signal(X, y)
        return float(base + silhouette_weight * sil if silhouette_weight != 0 else base + sil)

    if mode == "ilm_pca_unembed":
        base = ilm_pca_unembed_score(X, label_embs=label_embs, n_pcs=n_pcs, top_pc=top_pc)
        if fisher_weight or entropy_weight:
            fs = fisher_signal(X, y) if fisher_weight else 0.0
            ent = predictive_entropy_signal(X, y, seed=seed) if entropy_weight else 0.0
            return float(base + fisher_weight * fs + entropy_weight * ent)
        return base

    # Fallback to ilm_pca
    return ilm_pca_score(X, y, n_pcs=n_pcs, top_pc=top_pc)


def rerank_topk(candidates: List[int], scores: List[float]) -> int:
    """Return best layer from candidates using provided scores."""
    best_idx = int(np.argmax(scores))
    return int(candidates[best_idx])
