from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
import warnings
from sklearn.exceptions import ConvergenceWarning

from selection.data import resolve_dataset
from selection.qwen2_pc_patching import (
    ILMHeadPatchState,
    Qwen2HeadPatchContext,
    STATE,
    disable_patching,
    enable_patching,
    reset_collection,
    start_collection,
    stop_collection,
)

# Silence verbose warnings from Transformers and sklearn during analysis runs
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _train_val_split(y: np.ndarray, seed: int = 0, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split indices."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        cut = max(1, int(train_ratio * len(idx)))
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def _pool_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over non-pad tokens for final representation.
    hidden: (B, T, D), attention_mask: (B, T)
    """
    mask = attention_mask.float().unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def run_qwen2_head_pc_patching(
    llm_id: str,
    task: str,
    dataset: str,
    n_samples: int = 400,
    split: str = "validation",
    max_length: int = 128,
    device: str = "cuda",
    batch_size: int = 8,
    n_pcs: int = 4,
    top_pc: int = 2,
    seed: int = 2023,
) -> Dict[str, np.ndarray]:
    """
    Full ILM-style PC patching pipeline for Qwen2 attention heads.

    Steps:
      1) Sample (texts, labels) for the given task/dataset.
      2) Baseline pass with patched Qwen2 attention to COLLECT per-layer/head vectors.
      3) For each (layer, head), run PCA over collected vectors to obtain ILM PCs.
      4) Baseline pass (no patch) to get final-layer pooled representations and train a probe.
      5) For each (layer, head), enable PC patching and re-run forward to measure probe accuracy drop.

    Returns:
        dict with:
          - \"head_importance\": np.ndarray (num_layers, num_heads)
          - \"base_acc\": float
    """
    # 1) Resolve data
    texts, labels = resolve_dataset(task, dataset, n_samples=n_samples, seed=seed, split=split, stratified=True)
    y = np.asarray(labels)

    # 2) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
    model = AutoModel.from_pretrained(llm_id)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.float16)
    else:
        device = "cpu"
        model = model.to(device)
    model.eval()

    num_layers = len(model.layers)
    num_heads = model.layers[0].self_attn.num_heads

    with Qwen2HeadPatchContext():
        # 2) Baseline pass to COLLECT per-head vectors
        reset_collection()
        start_collection()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            # We only need hidden_states=False here; patched attention will collect internal head outputs.
            _ = model(**enc)

        stop_collection()

        # 3) Compute ILM PCs per (layer, head)
        head_pcs: Dict[Tuple[int, int], np.ndarray] = {}

        for layer_idx in range(num_layers):
            bucket = STATE.collected.get(layer_idx, [])
            if not bucket:
                continue
            # bucket: list of (num_heads, head_dim)
            H = np.stack(bucket, axis=0)  # (N, num_heads, head_dim)
            N, Hn, D = H.shape
            assert Hn == num_heads
            for h in range(num_heads):
                vecs = H[:, h, :]  # (N, D)
                if vecs.shape[0] < 10:
                    continue
                k = min(n_pcs, vecs.shape[0], vecs.shape[1])
                if k <= 0:
                    continue
                try:
                    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
                    S = pca.fit_transform(vecs)
                except Exception:
                    continue
                # Simple label-correlation to select top PCs
                from selection.scoring import label_correlation

                corrs = label_correlation(S, y[: S.shape[0]])
                if corrs.size == 0:
                    continue
                k_top = min(top_pc, corrs.size)
                idxs = np.argsort(-corrs)[:k_top]
                pcs = pca.components_[idxs]  # (k_top, D)
                head_pcs[(layer_idx, h)] = pcs.astype(np.float32)

        # Debug: report how many heads obtained ILM PCs
        if not head_pcs:
            print("[pc_patching] Warning: no ILM PCs were extracted for any head.")
        else:
            print(f"[pc_patching] Extracted ILM PCs for {len(head_pcs)} (layer, head) pairs.")

        # 4) Baseline final-layer representations + probe
        disable_patching()
        X_final: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]  # (B, T, D)
            pooled = _pool_last_hidden(last_hidden, enc["attention_mask"])  # (B, D)
            X_final.append(pooled.detach().float().cpu().numpy())

        X_final_np = np.vstack(X_final)
        # Train/val split for more sensitive patching evaluation
        train_idx, val_idx = _train_val_split(y, seed=seed, train_ratio=0.8)
        if len(val_idx) == 0 or len(train_idx) == 0:
            # Fallback: use all data as both train and val (less ideal)
            train_idx = np.arange(len(y))
            val_idx = np.arange(len(y))

        clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
        clf.fit(X_final_np[train_idx], y[train_idx])
        base_acc = float(accuracy_score(y[val_idx], clf.predict(X_final_np[val_idx])))

        # 5) Per-head patching evaluation
        head_importance = np.zeros((num_layers, num_heads), dtype=np.float32)

        for layer_idx in range(num_layers):
            for h in range(num_heads):
                pcs = head_pcs.get((layer_idx, h))
                if pcs is None:
                    continue
                # Use a larger lambda to make the effect more pronounced
                enable_patching(layer_idx, h, pcs, device=torch.device(device), lambda_scale=3.0)

                X_patched: List[np.ndarray] = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    enc = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out = model(**enc, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1]
                    pooled = _pool_last_hidden(last_hidden, enc["attention_mask"])
                    X_patched.append(pooled.detach().float().cpu().numpy())

                X_patched_np = np.vstack(X_patched)
                # Evaluate patch effect only on validation subset
                preds = clf.predict(X_patched_np[val_idx])
                acc_patched = float(accuracy_score(y[val_idx], preds))
                head_importance[layer_idx, h] = max(0.0, base_acc - acc_patched)

                disable_patching()

    return {
        "head_importance": head_importance,
        "base_acc": float(base_acc),
    }
