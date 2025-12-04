import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def _resolve_dataset(task: str, dataset: str, n_samples: int, seed: int = 42, split: str = 'validation',
                     stratified: bool = False) -> Tuple[List[str], List[int]]:
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    task = (task or "").lower()
    name = (dataset or "").lower()
    texts: List[str] = []
    labels: List[int] = []

    def take_subset(ds, text_key: str, label_key: str):
        nonlocal texts, labels
        n = min(n_samples, len(ds))
        if stratified:
            # balanced per-class sampling
            # build indices per class
            per_class = {}
            for i in range(len(ds)):
                item = ds[int(i)]
                c = int(item[label_key])
                per_class.setdefault(c, []).append(i)
            if len(per_class) == 0:
                return
            k = len(per_class)
            per_cls_n = max(1, n // k)
            for c, arr in per_class.items():
                take = min(per_cls_n, len(arr))
                idx = rng.choice(arr, size=take, replace=False)
                for j in idx:
                    item = ds[int(j)]
                    texts.append(str(item[text_key]))
                    labels.append(int(item[label_key]))
        else:
            idx = rng.choice(len(ds), size=n, replace=False)
            for i in idx:
                item = ds[int(i)]
                texts.append(str(item[text_key]))
                labels.append(int(item[label_key]))

    if name == "sst2":
        # Prefer SetFit/sst2; fallback to GLUE sst2 with proper text key
        try:
            ds = load_dataset("SetFit/sst2")
            chosen = ds.get(split if split in ds else "validation")
            take_subset(chosen, "text", "label")
        except Exception:
            ds = load_dataset("glue", "sst2")
            mapping = {"train":"train", "validation":"validation", "test":"test"}
            chosen = ds[mapping.get(split, "validation")]
            take_subset(chosen, "sentence", "label")
    elif name == "imdb":
        ds = load_dataset("imdb")
        take_subset(ds["test"], "text", "label")
    elif name == "cola":
        ds = load_dataset("nyu-mll/glue", "cola")
        take_subset(ds["validation"], "sentence", "label")
    elif name == "tweet_offensive":
        ds = load_dataset("cardiffnlp/tweet_eval", "offensive")
        take_subset(ds["validation"], "text", "label")
    elif name == "tweet_sentiment_binary":
        ds = load_dataset("tweet_eval", name="sentiment")
        val = ds["validation"]
        filtered = [ex for ex in val if int(ex["label"]) != 1]
        n = min(n_samples, len(filtered))
        idx = rng.choice(len(filtered), size=n, replace=False)
        for i in idx:
            item = filtered[int(i)]
            lab = 1 if int(item["label"]) == 2 else 0
            texts.append(str(item["text"]))
            labels.append(lab)
    elif name == "mnli":
        ds = load_dataset("glue", "mnli")
        # Map generic split to mnli split name
        split_map = {"train":"train", "validation":"validation_matched", "test":"validation_matched"}
        val = ds[split_map.get(split, "validation_matched")]
        n = min(n_samples, len(val))
        idx = rng.choice(len(val), size=n, replace=False)
        for i in idx:
            item = val[int(i)]
            texts.append(f"{item['premise']} [SEP] {item['hypothesis']}")
            labels.append(int(item["label"]))
    elif name == "snli":
        ds = load_dataset("snli")
        val = ds["validation"]
        n = min(n_samples, len(val))
        idx = rng.choice(len(val), size=n, replace=False)
        for i in idx:
            item = val[int(i)]
            if int(item["label"]) == -1:
                continue
            texts.append(f"{item['premise']} [SEP] {item['hypothesis']}")
            labels.append(int(item["label"]))
    return texts, labels


def _pool_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # Prefer first token; fallback to mean over unmasked tokens
    if hidden.size(1) > 1:
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        else:
            return hidden[:, 0]
    else:
        return hidden[:, 0]


@torch.no_grad()
def _get_llm_hidden_layers(llm_id: str, texts: List[str], device: str = "cuda", max_length: int = 128,
                           batch_size: int = 16, dtype: Optional[torch.dtype] = None, pooling: str = 'first',
                           l2_norm: bool = True) -> List[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
    model = AutoModel.from_pretrained(llm_id)
    if device == "cuda" and torch.cuda.is_available():
        if dtype is None:
            dtype = torch.float16
        model = model.to(device=device, dtype=dtype)
    else:
        device = "cpu"
        model = model.to(device)
    model.eval()

    all_layers: List[np.ndarray] = []
    # First pass to determine number of layers
    enc0 = tokenizer(texts[:1], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc0 = {k: v.to(device) for k, v in enc0.items()}
    out0 = model(**enc0, output_hidden_states=True)
    n_layers_total = len(out0.hidden_states) - 1  # exclude embeddings
    all_layers = [np.empty((0, out0.hidden_states[-1].size(-1)), dtype=np.float32) for _ in range(n_layers_total)]

    # Batch through texts
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = list(out.hidden_states)[1:]  # skip embeddings
        for li, h in enumerate(hs):
            if pooling == 'first':
                pooled = h[:, 0]
            else:
                pooled = _pool_hidden(h, enc.get("attention_mask"))
            pooled_np = pooled.detach().float().cpu().numpy()
            if l2_norm:
                norm = np.linalg.norm(pooled_np, axis=1, keepdims=True)
                pooled_np = pooled_np / np.maximum(norm, 1e-8)
            all_layers[li] = np.vstack([all_layers[li], pooled_np])
    return all_layers


def _pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    S = pca.fit_transform(X)
    comps = pca.components_
    return S, comps


def _label_correlation(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    else:
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


def _probe_effect(original: np.ndarray, patched: np.ndarray, y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(original, y)
    acc0 = accuracy_score(y, clf.predict(original))
    acc1 = accuracy_score(y, clf.predict(patched))
    return float(max(0.0, acc0 - acc1))


def _ilm_effect_for_layer(X: np.ndarray, y: np.ndarray, n_pcs: int, top_pc: int, lambdas: Tuple[float, ...]) -> float:
    scores, comps = _pca_scores(X, n_components=n_pcs)
    corrs = _label_correlation(scores, y)
    idx = np.argsort(-corrs)[:max(1, min(top_pc, n_pcs))]
    best = 0.0
    S_sel = scores[:, idx]
    U = comps[idx]  # (r, dim)
    recon = S_sel @ U  # (n, dim)
    for lam in lambdas:
        X_cf = X - lam * recon
        eff = _probe_effect(X, X_cf, y)
        best = max(best, eff)
    return best


def _fisher_class_separation(X: np.ndarray, y: np.ndarray) -> float:
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
    # inter-class: mean pairwise centroid distance^2
    inter_dists = []
    c_list = list(centroids.keys())
    for i in range(len(c_list)):
        for j in range(i + 1, len(c_list)):
            inter_dists.append(float(np.linalg.norm(centroids[c_list[i]] - centroids[c_list[j]]) ** 2))
    if len(inter_dists) == 0:
        return 0.0
    inter = float(np.mean(inter_dists))
    return inter / within


def _logit_probe_score(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
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
    clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class='auto')
    clf.fit(X[train_idx], y[train_idx])
    acc = accuracy_score(y[val_idx], clf.predict(X[val_idx]))
    return float(acc)


def _rerank_topk(candidates: List[int], scores: List[float]) -> int:
    """Return best layer from candidates using provided scores."""
    best_idx = int(np.argmax(scores))
    return int(candidates[best_idx])


def _compute_score_for_layer(X: np.ndarray, y: np.ndarray, mode: str, alpha: float, seed: int) -> float:
    """Unified scoring: probe / fisher / silhouette / mixed."""
    mode = (mode or "probe").lower()
    probe = _logit_probe_score(X, y, seed=seed)
    sil = 0.0
    try:
        sil = float(silhouette_score(X, y)) if len(np.unique(y)) > 1 and X.shape[0] > len(np.unique(y)) else 0.0
    except Exception:
        sil = 0.0
    if mode == "probe":
        return probe
    fisher = _fisher_class_separation(X, y)
    if mode == "fisher":
        return fisher
    if mode == "silhouette":
        return sil
    # mixed: combine probe, fisher, silhouette
    alpha = float(alpha)
    beta = float(getattr(_compute_score_for_layer, "_beta", 0.3))
    gamma = float(getattr(_compute_score_for_layer, "_gamma", 0.3))
    total = alpha + beta + gamma
    if total <= 0:
        return probe
    alpha /= total; beta /= total; gamma /= total
    return alpha * probe + beta * fisher + gamma * sil


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

    # Resolve huggingface model id for LLM
    from core.utils import get_huggingface_model_name
    llm_id = get_huggingface_model_name(llm_type)

    # Collect data
    texts, labels = _resolve_dataset(task, dataset, n_samples=n_samples, seed=seed, split=split, stratified=stratified)
    if len(texts) < 10:
        print("[selection] Not enough samples; defaulting to mid layer")
        cfg = AutoConfig.from_pretrained(llm_id)
        return max(0, cfg.num_hidden_layers // 2)

    # Collect pooled hidden per layer for LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (dtype_pref == 'fp16' and device == 'cuda') else torch.float32
    hidden_per_layer = _get_llm_hidden_layers(llm_id, texts, device=device, max_length=sel_max_len, batch_size=16, dtype=dtype, pooling=pooling, l2_norm=True)
    y = np.array(labels)

    depth_bias = float(getattr(args, "selection_depth_bias", 0.3))
    num_layers = len(hidden_per_layer)

    # 1st stage: coarse scoring (logit probe)
    effects: List[float] = []
    for Li, X in enumerate(hidden_per_layer):
        if stride > 1 and (Li % stride != 0):
            effects.append(0.0)
            continue
        try:
            eff = _compute_score_for_layer(X, y, mode=score_mode, alpha=score_alpha, seed=seed)
        except Exception as e:
            print(f"[selection] LLM layer {Li} scoring failed: {e}; using 0")
            eff = 0.0
        # depth prior: slightly favor shallower layers
        depth_weight = 1.0 - depth_bias * (Li / max(1, num_layers - 1))
        effects.append(eff * depth_weight)

    # Top-k rerank with same score (lightweight)
    k = int(getattr(args, "selection_topk_rerank", 3))
    eff_np = np.array(effects)
    top_idx = np.argsort(-eff_np)[:max(1, k)]
    top_scores = eff_np[top_idx]
    best_llm_layer = _rerank_topk(list(top_idx), list(top_scores))

    # Compute layer x PC correlation matrix for visualization (no extra forward pass)
    corr_mat = None
    try:
        corr_rows: List[np.ndarray] = []
        for X in hidden_per_layer:
            # Guard against tiny sample sizes
            pcs = min(n_pcs, X.shape[0], X.shape[1])
            if pcs <= 0:
                corr_rows.append(np.zeros((n_pcs,), dtype=np.float32))
                continue
            S, comps = _pca_scores(X, n_components=pcs)
            corrs = _label_correlation(S, y)
            # Pad/trim to n_pcs for consistent heatmap width
            row = np.zeros((n_pcs,), dtype=np.float32)
            row[:pcs] = corrs[:pcs]
            corr_rows.append(row)
        corr_mat = np.stack(corr_rows, axis=0)  # (layers, n_pcs)
    except Exception as _:
        corr_mat = None

    # Cache result
    out_dir = os.path.join(getattr(args, "result_path", "."), "layer_selection", task, dataset, slm_type, llm_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "selection.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": task,
            "dataset": dataset,
            "slm_type": slm_type,
            "llm_type": llm_type,
            "n_samples": n_samples,
            "n_pcs": n_pcs,
            "top_pc": top_pc,
            "effects": effects,
            "best_llm_layer": best_llm_layer,
            "seed": seed,
        }, f, indent=2)
    print(f"[selection] Saved selection to {out_path}")
    print(f"[selection] Best LLM layer: {best_llm_layer}")

    # Optional logging to W&B
    try:
        _log_selection_results(args, effects, best_llm_layer, out_path,
                               n_samples=n_samples, n_pcs=n_pcs, top_pc=top_pc,
                               corr_mat=corr_mat, hidden_per_layer=hidden_per_layer, labels=y)
    except Exception as e:
        print(f"[selection] Logging failed (non-fatal): {e}")

    # Optional archive images to .archive for quick sharing
    try:
        _save_selection_plots(args, effects, best_llm_layer, corr_mat=corr_mat,
                              hidden_per_layer=hidden_per_layer, labels=y)
    except Exception as e:
        print(f"[selection] Archiving plots failed (non-fatal): {e}")
    return best_llm_layer


def _log_selection_results(args,
                           effects: List[float],
                           best_llm_layer: int,
                           out_path: str,
                           n_samples: int,
                           n_pcs: int,
                           top_pc: int,
                           corr_mat: Optional[np.ndarray] = None,
                           hidden_per_layer: Optional[List[np.ndarray]] = None,
                           labels: Optional[np.ndarray] = None) -> None:
    """Log ILM selection summary to W&B (optional)."""
    # W&B
    if getattr(args, 'use_wandb', False) and getattr(args, 'log_selection', True):
        try:
            import wandb
            project = os.environ.get('WANDB_PROJECT') or getattr(args, 'proj_name', 'PiFi')
            entity = os.environ.get('WANDB_ENTITY')
            name = f"SELECTION - {args.task}/{args.task_dataset} - {args.model_type}/{args.llm_model}"
            init_kwargs = dict(project=project, name=name, config=dict(
                task=args.task, dataset=args.task_dataset,
                slm=args.model_type, llm=args.llm_model,
                n_samples=n_samples, n_pcs=n_pcs, top_pc=top_pc,
                seed=getattr(args, 'seed', 0)
            ), settings=wandb.Settings(save_code=False))
            if entity:
                init_kwargs['entity'] = entity
            run = wandb.init(**init_kwargs)

            # Log table of effects
            tbl = wandb.Table(data=[[int(i), float(v)] for i, v in enumerate(effects)], columns=['layer', 'effect'])
            wandb.log({'selection/effects_table': tbl})

            # Plot images if available
            imgs = []
            if plt is not None:
                # line
                fig_line, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(list(range(len(effects))), effects, marker='o')
                ax.axvline(best_llm_layer, color='r', linestyle='--', label=f'best={best_llm_layer}')
                ax.set_title('ILM Effect per Layer (line)')
                ax.set_xlabel('Layer'); ax.set_ylabel('Effect'); ax.legend(loc='upper right')
                fig_line.tight_layout()
                imgs.append(wandb.Image(fig_line, caption='effects_line'))
                plt.close(fig_line)
            if plt is not None and sns is not None:
                fig_hm, ax = plt.subplots(figsize=(10, 2.0))
                sns.heatmap([effects], cmap='viridis', cbar=True, xticklabels=list(range(len(effects))), yticklabels=['effect'], ax=ax)
                ax.axvline(best_llm_layer + 0.5, color='r', linestyle='--')
                ax.set_title('ILM Effect per Layer (heatmap)')
                fig_hm.tight_layout()
                imgs.append(wandb.Image(fig_hm, caption='effect_heatmap'))
                plt.close(fig_hm)
            # Correlation heatmap image
            if corr_mat is not None and plt is not None and sns is not None:
                fig_corr, ax = plt.subplots(figsize=(max(6, n_pcs/2), max(3, corr_mat.shape[0]/4)))
                sns.heatmap(corr_mat, cmap='magma', cbar=True,
                            xticklabels=list(range(n_pcs)),
                            yticklabels=list(range(corr_mat.shape[0])), ax=ax)
                ax.set_title('ILM Layer x PC Label-Correlation')
                ax.set_xlabel('PC'); ax.set_ylabel('Layer')
                fig_corr.tight_layout()
                imgs.append(wandb.Image(fig_corr, caption='corr_heatmap'))
                plt.close(fig_corr)

            # PCA scatter plots for selected layers
            if getattr(args, 'log_selection_pca', True) and hidden_per_layer is not None and labels is not None and plt is not None:
                sel_layers = _select_plot_layers(len(hidden_per_layer), best_llm_layer,
                                                 getattr(args, 'selection_plot_layers', 'best,first,mid,last'),
                                                 getattr(args, 'selection_plot_max_layers', 6))
                for li in sel_layers:
                    X = hidden_per_layer[li]
                    pcs = min(2, X.shape[0], X.shape[1])
                    if pcs < 2:
                        continue
                    S2, _ = _pca_scores(X, n_components=2)
                    fig_sc, ax = plt.subplots(figsize=(4, 3))
                    labs = np.array(labels)
                    uniq = np.unique(labs)
                    for c in uniq:
                        idx = labs == c
                        ax.scatter(S2[idx, 0], S2[idx, 1], s=8, alpha=0.7, label=str(int(c)))
                    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
                    ax.set_title(f'PCA Scatter - Layer {li}')
                    ax.legend(markerscale=2, fontsize=7, loc='best', frameon=False)
                    fig_sc.tight_layout()
                    imgs.append(wandb.Image(fig_sc, caption=f'pca_scatter_layer_{li}'))
                    plt.close(fig_sc)

            if imgs:
                wandb.log({'selection/plots': imgs})

            # Attach selection.json as artifact
            try:
                art = wandb.Artifact(f"selection_{args.task}_{args.task_dataset}_{args.model_type}_{args.llm_model}", type='selection')
                art.add_file(out_path)
                run.log_artifact(art)
            except Exception as e2:
                print(f"[selection] W&B artifact error: {e2}")

            run.finish()
        except Exception as e:
            print(f"[selection] W&B logging error: {e}")


def _save_selection_plots(args, effects: List[float], best_llm_layer: int, corr_mat: Optional[np.ndarray] = None,
                          hidden_per_layer: Optional[List[np.ndarray]] = None, labels: Optional[np.ndarray] = None) -> None:
    """Save selection plots as images under .archive for quick sharing."""
    if plt is None:
        return
    base = os.path.join('.archive', 'selection_plots', args.task, args.task_dataset, args.model_type, args.llm_model)
    os.makedirs(base, exist_ok=True)
    ts = ''  # could add timestamp if desired

    # Line plot
    fig_line, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(list(range(len(effects))), effects, marker='o')
    ax.axvline(best_llm_layer, color='r', linestyle='--', label=f'best={best_llm_layer}')
    ax.set_title('ILM Effect per Layer (line)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Effect'); ax.legend(loc='upper right')
    fig_line.tight_layout()
    path_line = os.path.join(base, f'effect_line{ts}.png')
    fig_line.savefig(path_line, dpi=150)
    plt.close(fig_line)

    # Heatmap of effects
    if sns is not None:
        fig_hm, ax = plt.subplots(figsize=(10, 2.0))
        sns.heatmap([effects], cmap='viridis', cbar=True, xticklabels=list(range(len(effects))), yticklabels=['effect'], ax=ax)
        ax.axvline(best_llm_layer + 0.5, color='r', linestyle='--')
        ax.set_title('ILM Effect per Layer (heatmap)')
        fig_hm.tight_layout()
        path_hm = os.path.join(base, f'effect_heatmap{ts}.png')
        fig_hm.savefig(path_hm, dpi=150)
        plt.close(fig_hm)

    # Correlation heatmap (layers x PCs)
    if corr_mat is not None and sns is not None:
        fig_corr, ax = plt.subplots(figsize=(max(6, corr_mat.shape[1]/2), max(3, corr_mat.shape[0]/4)))
        sns.heatmap(corr_mat, cmap='magma', cbar=True,
                    xticklabels=list(range(corr_mat.shape[1])),
                    yticklabels=list(range(corr_mat.shape[0])), ax=ax)
        ax.set_title('ILM Layer x PC Label-Correlation')
        ax.set_xlabel('PC'); ax.set_ylabel('Layer')
        fig_corr.tight_layout()
        path_corr = os.path.join(base, f'corr_heatmap{ts}.png')
        fig_corr.savefig(path_corr, dpi=150)
        plt.close(fig_corr)

    # PCA scatter plots for selected layers
    if getattr(args, 'log_selection_pca', True) and hidden_per_layer is not None and labels is not None and plt is not None:
        sel_layers = _select_plot_layers(len(hidden_per_layer), best_llm_layer,
                                         getattr(args, 'selection_plot_layers', 'best,first,mid,last'),
                                         getattr(args, 'selection_plot_max_layers', 6))
        for li in sel_layers:
            X = hidden_per_layer[li]
            pcs = min(2, X.shape[0], X.shape[1])
            if pcs < 2:
                continue
            S2, _ = _pca_scores(X, n_components=2)
            fig_sc, ax = plt.subplots(figsize=(4, 3))
            labs = np.array(labels)
            uniq = np.unique(labs)
            for c in uniq:
                idx = labs == c
                ax.scatter(S2[idx, 0], S2[idx, 1], s=8, alpha=0.7, label=str(int(c)))
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.set_title(f'PCA Scatter - Layer {li}')
            ax.legend(markerscale=2, fontsize=7, loc='best', frameon=False)
            fig_sc.tight_layout()
            path_sc = os.path.join(base, f'pca_scatter_layer_{li}{ts}.png')
            fig_sc.savefig(path_sc, dpi=150)
            plt.close(fig_sc)


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
