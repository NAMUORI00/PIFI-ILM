from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score


def pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    S = pca.fit_transform(X)
    comps = pca.components_
    return S, comps


def label_correlation(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def fisher_class_separation(X: np.ndarray, y: np.ndarray) -> float:
    """Class separation score: inter-centroid distance / intra-class variance."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or X.shape[0] == 0:
        return 0.0
    centroids = {}
    within = 0.0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        var = ((Xc - mu) ** 2).sum(axis=1).mean()
        within += float(var)
    within = max(within, 1e-6)
    if len(centroids) < 2:
        return 0.0
    inter_dists = []
    c_list = list(centroids.keys())
    for i in range(len(c_list)):
        for j in range(i + 1, len(c_list)):
            inter_dists.append(float(np.linalg.norm(centroids[c_list[i]] - centroids[c_list[j]]) ** 2))
    if not inter_dists:
        return 0.0
    inter = float(np.mean(inter_dists))
    return inter / within


def logit_probe_score(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Lightweight linear probe accuracy as a proxy for downstream performance."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or X.shape[0] < 10:
        return 0.0
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])
    if len(train_idx) == 0 or len(val_idx) == 0:
        return 0.0
    clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto")
    clf.fit(X[train_idx], y[train_idx])
    acc = accuracy_score(y[val_idx], clf.predict(X[val_idx]))
    return float(acc)


def robust_probe_score(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_seeds: int = 3,
    use_selectivity: bool = True,
) -> Tuple[float, float]:
    """K-fold + multi-seed probe with optional selectivity."""
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or X.shape[0] < 20:
        return 0.0, 0.0

    all_accs: List[float] = []
    all_random_accs: List[float] = []
    all_confidences: List[float] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for train_idx, val_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
            clf.fit(X[train_idx], y[train_idx])

            probs = clf.predict_proba(X[val_idx])
            preds = clf.predict(X[val_idx])
            acc = accuracy_score(y[val_idx], preds)
            conf = float(np.mean(np.max(probs, axis=1)))

            all_accs.append(acc)
            all_confidences.append(conf)

            if use_selectivity:
                y_train_random = rng.permutation(y[train_idx])
                clf_rand = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
                try:
                    clf_rand.fit(X[train_idx], y_train_random)
                    y_val_random = rng.permutation(y[val_idx])
                    acc_rand = accuracy_score(y_val_random, clf_rand.predict(X[val_idx]))
                    all_random_accs.append(acc_rand)
                except Exception:
                    pass

    mean_acc = float(np.mean(all_accs)) if all_accs else 0.0
    mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

    if use_selectivity and all_random_accs:
        mean_random = float(np.mean(all_random_accs))
        selectivity = max(0.0, mean_acc - mean_random)
        return selectivity, mean_conf

    return mean_acc, mean_conf


def compute_score_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
    n_folds: int = 5,
    n_seeds: int = 3,
    use_confidence_weight: bool = False,
) -> float:
    """
    Unified scoring: probe / fisher / silhouette / mixed / robust.
    """
    mode = (mode or "probe").lower()

    if mode == "robust":
        score, conf = robust_probe_score(X, y, n_folds=n_folds, n_seeds=n_seeds, use_selectivity=True)
        if use_confidence_weight:
            return score * (0.5 + 0.5 * conf)
        return score

    probe = logit_probe_score(X, y, seed=seed)
    sil = 0.0
    try:
        sil = float(silhouette_score(X, y)) if len(np.unique(y)) > 1 and X.shape[0] > len(np.unique(y)) else 0.0
    except Exception:
        sil = 0.0
    if mode == "probe":
        return probe

    fisher = fisher_class_separation(X, y)
    if mode == "fisher":
        return fisher
    if mode == "silhouette":
        return sil

    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    total = alpha + beta + gamma
    if total <= 0:
        return probe
    alpha /= total
    beta /= total
    gamma /= total
    return alpha * probe + beta * fisher + gamma * sil


def rerank_topk(candidates: List[int], scores: List[float]) -> int:
    """Return best layer from candidates using provided scores."""
    best_idx = int(np.argmax(scores))
    return int(candidates[best_idx])

