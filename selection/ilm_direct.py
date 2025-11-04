import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _resolve_dataset(task: str, dataset: str, n_samples: int, seed: int = 42) -> Tuple[List[str], List[int]]:
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    task = (task or "").lower()
    name = (dataset or "").lower()
    texts: List[str] = []
    labels: List[int] = []

    def take_subset(ds, text_key: str, label_key: str):
        nonlocal texts, labels
        n = min(n_samples, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        for i in idx:
            item = ds[int(i)]
            texts.append(str(item[text_key]))
            labels.append(int(item[label_key]))

    if name == "sst2":
        ds = load_dataset("SetFit/sst2")
        take_subset(ds["validation"], "text", "label")
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
        val = ds["validation_matched"]
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
def _get_llm_hidden_layers(llm_id: str, texts: List[str], device: str = "cuda", max_length: int = 128, batch_size: int = 16, dtype: Optional[torch.dtype] = None) -> List[np.ndarray]:
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
    hidden_states_sample = None
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
            pooled = _pool_hidden(h, enc.get("attention_mask"))
            pooled_np = pooled.detach().float().cpu().numpy()
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
    lambdas = tuple(getattr(args, "selection_patch_lambda", (0.5, 1.0))) if isinstance(getattr(args, "selection_patch_lambda", (0.5, 1.0)), (list, tuple)) else (0.5, 1.0)
    stride = int(getattr(args, "selection_layer_stride", 1))

    # Resolve huggingface model id for LLM
    from Classification.utils.utils import get_huggingface_model_name
    llm_id = get_huggingface_model_name(llm_type)

    # Collect data
    texts, labels = _resolve_dataset(task, dataset, n_samples=n_samples, seed=seed)
    if len(texts) < 10:
        print("[selection] Not enough samples; defaulting to mid layer")
        cfg = AutoConfig.from_pretrained(llm_id)
        return max(0, cfg.num_hidden_layers // 2)

    # Collect pooled hidden per layer for LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_per_layer = _get_llm_hidden_layers(llm_id, texts, device=device, max_length=128, batch_size=16)
    y = np.array(labels)

    effects: List[float] = []
    for Li, X in enumerate(hidden_per_layer):
        if stride > 1 and (Li % stride != 0):
            effects.append(0.0)
            continue
        try:
            eff = _ilm_effect_for_layer(X, y, n_pcs=n_pcs, top_pc=top_pc, lambdas=lambdas)
        except Exception as e:
            print(f"[selection] LLM layer {Li} scoring failed: {e}; using 0")
            eff = 0.0
        effects.append(eff)

    best_llm_layer = int(np.argmax(np.array(effects)))

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
    return best_llm_layer

