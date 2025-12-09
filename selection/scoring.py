"""Layer selection scoring functions.

Only two methods are supported:
1. ILM-PCA: PC-label correlation
2. MDL: Minimum Description Length
"""
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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


def online_coding_mdl(
    X: np.ndarray, y: np.ndarray, n_portions: int = 10, seed: int = 0
) -> float:
    """
    Online Coding MDL score (Voita & Titov 2020).

    Returns average code length (lower is better).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n = len(y)

    if n < 20:
        return float("inf")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    portion_size = n // n_portions
    if portion_size < 2:
        return float("inf")

    total_codelength = 0.0
    n_classes = len(np.unique(y))

    # First portion: uniform code (no model)
    first_portion_cost = portion_size * np.log2(max(n_classes, 2))
    total_codelength += first_portion_cost

    # Online coding for remaining portions
    for i in range(1, n_portions):
        train_end = i * portion_size
        test_start = train_end
        test_end = (i + 1) * portion_size if i < n_portions - 1 else n

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if len(X_test) == 0:
            continue

        try:
            clf = LogisticRegression(max_iter=300, random_state=seed, n_jobs=1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)

            for j, true_label in enumerate(y_test):
                label_idx = list(clf.classes_).index(true_label)
                prob = max(probs[j, label_idx], 1e-10)
                total_codelength += -np.log2(prob)
        except Exception:
            total_codelength += len(y_test) * np.log2(max(n_classes, 2))

    return total_codelength / n


def compute_score_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "ilm_pca",
    n_pcs: int = 16,
    top_pc: int = 5,
    mdl_n_portions: int = 10,
    seed: int = 0,
) -> float:
    """
    Compute layer selection score.

    Returns higher-is-better score for both modes (MDL is negated).
    """
    mode = (mode or "ilm_pca").lower()

    if mode == "ilm_pca":
        return ilm_pca_score(X, y, n_pcs=n_pcs, top_pc=top_pc)

    if mode == "mdl":
        mdl = online_coding_mdl(X, y, n_portions=mdl_n_portions, seed=seed)
        return -mdl

    # Fallback to ilm_pca
    return ilm_pca_score(X, y, n_pcs=n_pcs, top_pc=top_pc)


def rerank_topk(candidates: List[int], scores: List[float]) -> int:
    """Return best layer from candidates using provided scores."""
    best_idx = int(np.argmax(scores))
    return int(candidates[best_idx])
