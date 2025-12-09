"""Qwen2-focused layer selection following the 3-step PCL → PC patching plan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from collections import Counter
import json
import os
import re

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

try:
    from core.utils import get_huggingface_model_name
except Exception:  # fallback when core package is unavailable (e.g., standalone selection)
    try:
        from Classification.utils.utils import get_huggingface_model_name  # type: ignore
    except Exception:
        from TextualEntailment.utils.utils import get_huggingface_model_name  # type: ignore
from selection.data import resolve_dataset
from selection.label_embeddings import get_label_token_hidden_per_layer
from selection.prompts import build_label_prompts
from selection.qwen2_pc_patching import Qwen2HeadPatchContext, disable_patching, enable_patching
from selection.scoring import label_correlation

# Lightweight stopword list to avoid pulling external deps
STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "to",
    "and",
    "or",
    "of",
    "for",
    "with",
    "at",
    "is",
    "are",
    "was",
    "were",
    "it",
    "this",
    "that",
    "be",
    "as",
    "by",
    "from",
    "not",
    "but",
    "we",
    "you",
    "they",
    "their",
    "our",
    "your",
}


@dataclass
class AbstractLayerResult:
    layer: int
    scores: List[float]
    anchor_pc: Optional[np.ndarray]
    keywords: List[str]
    match_stats: Dict[str, int]
    hidden_per_layer: List[np.ndarray]
    prompts: List[str]
    label_token_strs: List[List[str]]
    labels: np.ndarray
    tokenizer: AutoTokenizer
    model: AutoModel
    per_layer_signals: List[Dict[str, float]]


@dataclass
class ApplyLayerResult:
    layer: int
    head_effects: np.ndarray
    base_acc: float
    per_layer_probe_acc: Optional[List[float]] = None  # Probe accuracy for each layer


def _train_val_split(y: np.ndarray, seed: int = 0, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
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
    if len(train_idx) == 0 or len(val_idx) == 0:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        cut = max(1, int(train_ratio * len(idx)))
        train_idx, val_idx = idx[:cut], idx[cut:]
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def _default_max_length(dataset: str, provided: int, max_seq_len: int) -> int:
    if provided > 0:
        return provided
    if max_seq_len > 0:
        return max_seq_len
    name = (dataset or "").lower()
    if name in {"imdb"}:
        return 256
    if name in {"mnli", "snli"}:
        return 192
    if name in {"tweet_offensive", "tweet_sentiment_binary"}:
        return 128
    return 128


def _normalize_keywords(cands: Sequence[str], top_k: int) -> List[str]:
    uniq = []
    for w in cands:
        w = w.strip().lower()
        if not w or w in STOPWORDS:
            continue
        if not re.match(r"^[a-z][a-z0-9_-]{2,}$", w):
            continue
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= top_k:
            break
    return uniq


def _extract_keywords_tfidf(texts: Sequence[str], top_k: int, seed: int) -> List[str]:
    try:
        vec = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        idxs = np.argsort(-scores)[: top_k * 3]
        vocab = np.array(vec.get_feature_names_out())[idxs]
        return _normalize_keywords(vocab.tolist(), top_k)
    except Exception:
        # Fallback to frequency if TF-IDF fails
        counter: Counter[str] = Counter()
        rng = np.random.default_rng(seed)
        for t in texts:
            tokens = re.findall(r"[A-Za-z]{3,}", t.lower())
            tokens = [tok for tok in tokens if tok not in STOPWORDS]
            rng.shuffle(tokens)
            counter.update(tokens)
        if not counter:
            return []
        return _normalize_keywords([w for w, _ in counter.most_common(top_k * 3)], top_k)


def _extract_keywords_llm(
    texts: Sequence[str],
    llm_id: str,
    tokenizer: AutoTokenizer,
    device: str,
    top_k: int,
    sample_limit: int,
) -> Optional[List[str]]:
    """Ask the LLM itself to propose task keywords; best-effort with safe fallbacks."""
    try:
        subtexts = list(texts)[: max(1, sample_limit)]
        prompt_lines = ["Extract the top task keywords (comma-separated, single words) for these examples:"]
        for idx, t in enumerate(subtexts[:6]):
            prompt_lines.append(f"{idx+1}. {t[:200]}")
        prompt_lines.append("Keywords:")
        prompt = "\n".join(prompt_lines)
        gen_model = AutoModelForCausalLM.from_pretrained(llm_id)
        gen_model = gen_model.to(device)
        gen_model.eval()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen_cfg = GenerationConfig(
            max_new_tokens=64,
            do_sample=False,
            temperature=0.7,
            num_beams=1,
            pad_token_id=pad_id,
        )
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = gen_model.generate(**inputs, generation_config=gen_cfg)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Take trailing segment after "Keywords:" if present
        if "Keywords:" in decoded:
            decoded = decoded.split("Keywords:", 1)[1]
        parts = re.split(r"[,\n]", decoded)
        kws = _normalize_keywords(parts, top_k)
        return kws if kws else None
    except Exception:
        return None


def _extract_task_keywords(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    llm_id: str,
    source: str,
    top_k: int,
    seed: int,
    sample_limit: int,
    device: str,
) -> List[str]:
    source = (source or "tfidf").lower()
    sample_texts = list(texts)
    eff_limit = sample_limit if sample_limit > 0 else len(sample_texts)
    if len(sample_texts) > eff_limit > 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sample_texts), size=eff_limit, replace=False)
        sample_texts = [sample_texts[int(i)] for i in idx]
    if source == "llm":
        kw = _extract_keywords_llm(sample_texts, llm_id, tokenizer, device, top_k=top_k, sample_limit=eff_limit)
        if kw:
            return kw
        # fall back to tfidf if LLM fails
    return _extract_keywords_tfidf(sample_texts, top_k=top_k, seed=seed)


def _keyword_embeddings(keywords: List[str], tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    if not keywords:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    weight = model.get_input_embeddings().weight
    embs: List[np.ndarray] = []
    for kw in keywords:
        ids = tokenizer(kw, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        vec = weight[ids, :].mean(dim=0)
        embs.append(vec.detach().float().cpu().numpy())
    if not embs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.stack(embs, axis=0)


def _pcl_keyword_score(
    X: np.ndarray,
    y: np.ndarray,
    keyword_embs: np.ndarray,
    n_pcs: int,
    top_pc: int,
    keyword_weight: float,
) -> Tuple[float, Optional[np.ndarray], Dict[str, float]]:
    """Compute PCL-inspired score + return the keyword-aligned PC."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] == 0:
        return 0.0, None, {"keyword_signal": 0.0, "corr_signal": 0.0}

    pcs = min(n_pcs, X.shape[0], X.shape[1])
    if pcs <= 0:
        return 0.0, None, {"keyword_signal": 0.0, "corr_signal": 0.0}

    try:
        pca = PCA(n_components=pcs, random_state=0)
        S = pca.fit_transform(X)
    except Exception:
        return 0.0, None, {"keyword_signal": 0.0, "corr_signal": 0.0}

    comps = pca.components_  # (pcs, hidden_dim)
    corrs = label_correlation(S, y)
    corr_signal = 0.0
    if corrs.size > 0:
        k = min(top_pc, corrs.size)
        corr_signal = float(np.mean(np.sort(corrs)[-k:]))

    keyword_signal = 0.0
    best_pc = None
    if keyword_embs.size > 0:
        comp_norm = comps / np.maximum(np.linalg.norm(comps, axis=1, keepdims=True), 1e-8)
        kw_norm = keyword_embs / np.maximum(np.linalg.norm(keyword_embs, axis=1, keepdims=True), 1e-8)
        sim = comp_norm @ kw_norm.T  # (pcs, num_keywords)
        keyword_signal = float(np.max(sim)) if sim.size > 0 else 0.0
        if sim.size > 0:
            best_idx = int(np.unravel_index(np.argmax(sim), sim.shape)[0])
            best_pc = comps[best_idx]
    if best_pc is None and comps.shape[0] > 0:
        best_pc = comps[0]

    score = keyword_weight * keyword_signal + (1.0 - keyword_weight) * corr_signal
    return float(score), best_pc.astype(np.float32) if best_pc is not None else None, {
        "keyword_signal": keyword_signal,
        "corr_signal": corr_signal,
    }


def _reshape_anchor_for_heads(anchor_pc: Optional[np.ndarray], model: AutoModel) -> Optional[np.ndarray]:
    if anchor_pc is None:
        return None
    if not hasattr(model, "layers") or len(model.layers) == 0:
        return None
    attn = model.layers[0].self_attn
    num_heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    target = num_heads * head_dim
    vec = np.asarray(anchor_pc, dtype=np.float32).flatten()
    if vec.size < target:
        reps = int(np.ceil(target / max(1, vec.size)))
        vec = np.tile(vec, reps)
    vec = vec[:target]
    return vec.reshape(num_heads, head_dim)


def _prepare_model(llm_id: str, device: str, dtype: torch.dtype) -> Tuple[AutoTokenizer, AutoModel, str]:
    tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
    model = AutoModel.from_pretrained(llm_id)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device=device, dtype=dtype)
    else:
        device = "cpu"
        model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return tokenizer, model, device


def identify_abstract_layer(
    llm_id: str,
    texts: List[str],
    labels: List[int],
    n_pcs: int,
    top_pc: int,
    k_shot: int,
    max_length: int,
    device: str,
    dtype: torch.dtype,
    keyword_top_k: int,
    keyword_source: str,
    keyword_llm_id: Optional[str],
    keyword_sample_limit: int,
    keyword_weight: float,
    seed: int,
) -> AbstractLayerResult:
    prompts, label_token_strs, prompt_labels = build_label_prompts(texts, labels, k_shot=k_shot)
    tokenizer, model, device = _prepare_model(llm_id, device, dtype)

    hidden_per_layer, _, match_stats = get_label_token_hidden_per_layer(
        llm_id,
        prompts,
        label_token_strs,
        device=device,
        max_length=max_length,
        batch_size=8,
        dtype=dtype,
        return_label_embs=False,
        tokenizer=tokenizer,
        model=model,
    )
    kw_llm_id = keyword_llm_id or llm_id
    kw_tokenizer = tokenizer if kw_llm_id == llm_id else AutoTokenizer.from_pretrained(kw_llm_id, use_fast=True)
    keywords = _extract_task_keywords(
        texts,
        tokenizer=kw_tokenizer,
        llm_id=kw_llm_id,
        source=keyword_source,
        top_k=keyword_top_k,
        seed=seed,
        sample_limit=keyword_sample_limit,
        device=device,
    )
    keyword_embs = _keyword_embeddings(keywords, tokenizer, model)
    y = np.asarray(prompt_labels, dtype=int)

    scores: List[float] = []
    pcs: List[Optional[np.ndarray]] = []
    per_layer_signals: List[Dict[str, float]] = []
    for layer_idx, X in enumerate(hidden_per_layer):
        s, pc, sig = _pcl_keyword_score(
            X,
            y,
            keyword_embs,
            n_pcs=n_pcs,
            top_pc=top_pc,
            keyword_weight=keyword_weight,
        )
        scores.append(s)
        pcs.append(pc)
        per_layer_signals.append(sig)

    if not scores:
        best_layer = -1
        anchor_pc = None
    else:
        best_layer = int(np.argmax(scores))
        anchor_pc = pcs[best_layer] if 0 <= best_layer < len(pcs) else None
        if np.max(scores) <= 0:
            best_layer = max(0, len(hidden_per_layer) // 2)
            anchor_pc = pcs[best_layer] if 0 <= best_layer < len(pcs) else anchor_pc

    return AbstractLayerResult(
        layer=best_layer,
        scores=scores,
        anchor_pc=anchor_pc,
        keywords=keywords,
        match_stats=match_stats,
        hidden_per_layer=hidden_per_layer,
        prompts=prompts,
        label_token_strs=label_token_strs,
        labels=y,
        tokenizer=tokenizer,
        model=model,
        per_layer_signals=per_layer_signals,
    )


def identify_apply_layer(
    llm_id: str,
    abstract: AbstractLayerResult,
    lambda_scale: float,
    max_length: int,
    batch_size: int,
    seed: int,
    val_limit: Optional[int] = None,
    max_layers: int = 0,
    max_heads: int = 0,
) -> ApplyLayerResult:
    model = abstract.model
    tokenizer = abstract.tokenizer
    torch_device = next(model.parameters()).device
    device_str = "cuda" if torch_device.type == "cuda" else "cpu"
    dtype = next(model.parameters()).dtype

    # Base probe on final-layer label representations
    X_final = abstract.hidden_per_layer[-1]
    y = abstract.labels
    if X_final.shape[0] < 10:
        return ApplyLayerResult(layer=-1, head_effects=np.zeros((0, 0), dtype=np.float32), base_acc=0.0, per_layer_probe_acc=None)

    train_idx, val_idx = _train_val_split(y, seed=seed, train_ratio=0.8)
    if val_limit and len(val_idx) > val_limit:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(val_idx), size=val_limit, replace=False)
        val_idx = val_idx[chosen]
    if len(val_idx) == 0:
        return ApplyLayerResult(layer=-1, head_effects=np.zeros((0, 0), dtype=np.float32), base_acc=0.0, per_layer_probe_acc=None)

    # Compute per-layer probe accuracy
    per_layer_probe_acc: List[float] = []
    for layer_idx, X_layer in enumerate(abstract.hidden_per_layer):
        try:
            layer_clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
            layer_clf.fit(X_layer[train_idx], y[train_idx])
            layer_acc = float(accuracy_score(y[val_idx], layer_clf.predict(X_layer[val_idx])))
            per_layer_probe_acc.append(layer_acc)
        except Exception:
            per_layer_probe_acc.append(0.0)

    clf = LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto", random_state=seed)
    try:
        clf.fit(X_final[train_idx], y[train_idx])
        # Store base probabilities on the validation subset for KL/Δlogit scoring
        base_proba = clf.predict_proba(X_final[val_idx])  # (N_val, K)
        base_acc = float(accuracy_score(y[val_idx], clf.predict(X_final[val_idx])))
    except Exception:
        return ApplyLayerResult(layer=-1, head_effects=np.zeros((0, 0), dtype=np.float32), base_acc=0.0, per_layer_probe_acc=per_layer_probe_acc)

    # Limit patching to the validation subset to keep runtime reasonable
    val_prompts = [abstract.prompts[i] for i in val_idx]
    val_label_token_strs = [abstract.label_token_strs[i] for i in val_idx]

    num_layers = len(model.layers) if hasattr(model, "layers") else 0
    num_heads = int(model.layers[0].self_attn.num_heads) if num_layers > 0 else 0
    if max_layers > 0:
        num_layers = min(num_layers, max_layers)
    if max_heads > 0:
        num_heads = min(num_heads, max_heads)
    head_vectors = _reshape_anchor_for_heads(abstract.anchor_pc, model)
    if head_vectors is None or num_layers == 0 or num_heads == 0:
        return ApplyLayerResult(
            layer=-1,
            head_effects=np.zeros((num_layers, num_heads), dtype=np.float32),
            base_acc=base_acc,
            per_layer_probe_acc=per_layer_probe_acc,
        )

    head_effects = np.zeros((num_layers, num_heads), dtype=np.float32)
    with Qwen2HeadPatchContext():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                pcs = head_vectors[head_idx : head_idx + 1]
                enable_patching(
                    layer_idx,
                    head_idx,
                    pcs,
                    device=torch_device,
                    lambda_scale=lambda_scale,
                )
                try:
                    patched_layers, _, _ = get_label_token_hidden_per_layer(
                        llm_id,
                        val_prompts,
                        val_label_token_strs,
                        device=device_str,
                        max_length=max_length,
                        batch_size=batch_size,
                        dtype=dtype,
                        return_label_embs=False,
                        tokenizer=tokenizer,
                        model=model,
                    )
                    X_val_patched = patched_layers[-1]
                    proba_patched = clf.predict_proba(X_val_patched)
                    # Clamp to avoid log(0) in KL
                    eps = 1e-8
                    p = base_proba
                    q = proba_patched
                    p = p.clip(eps, 1.0)
                    q = q.clip(eps, 1.0)
                    p /= p.sum(axis=1, keepdims=True)
                    q /= q.sum(axis=1, keepdims=True)
                    # D_KL(p || q) averaged over validation set
                    kl = float((p * (np.log(p) - np.log(q))).sum(axis=1).mean())
                    effect = max(0.0, kl)
                except Exception as e:
                    if layer_idx == 0 and head_idx == 0:
                        import traceback
                        print(f"[DEBUG] Exception in head patching: {type(e).__name__}: {e}")
                        traceback.print_exc()
                    effect = 0.0
                head_effects[layer_idx, head_idx] = effect
                disable_patching()

    apply_layer = int(np.argmax(head_effects.sum(axis=1))) if head_effects.size > 0 else -1
    return ApplyLayerResult(layer=apply_layer, head_effects=head_effects, base_acc=base_acc, per_layer_probe_acc=per_layer_probe_acc)


def auto_select_layer(args) -> int:
    """Run the 3-step PCL + patching pipeline and return the chosen LLM layer index."""
    task = getattr(args, "task", "classification")
    dataset = getattr(args, "task_dataset", "sst2")
    llm_type = getattr(args, "llm_model", None) or getattr(args, "llm", None) or "qwen2"
    llm_id = get_huggingface_model_name(llm_type)

    n_samples = int(getattr(args, "selection_samples", 200))
    n_pcs = int(getattr(args, "selection_pcs", 16))
    top_pc = int(getattr(args, "selection_top_pc", 5))
    k_shot = int(getattr(args, "selection_k_shot", 1))
    max_length = _default_max_length(
        dataset,
        int(getattr(args, "selection_max_length", 0)),
        int(getattr(args, "max_seq_len", 0)),
    )
    keyword_top_k = int(getattr(args, "selection_keyword_top_k", 12))
    keyword_source = getattr(args, "selection_keyword_source", "tfidf")
    keyword_llm_id = getattr(args, "selection_keyword_llm_id", None)
    keyword_sample_limit = int(getattr(args, "selection_keyword_samples", 64))
    keyword_weight = float(getattr(args, "selection_keyword_weight", 0.65))
    lambda_scale = float(getattr(args, "selection_lambda_scale", 3.0))
    patch_val_limit = int(getattr(args, "selection_patch_eval_samples", 0))
    patch_batch_size = int(getattr(args, "selection_patch_batch_size", 8))
    multi_layer_span = int(getattr(args, "selection_multi_layer_span", 0))
    seed = int(getattr(args, "seed", 2023))
    sel_device = getattr(args, "device", "cuda")
    dtype_pref = getattr(args, "selection_dtype", "fp16")
    dtype = torch.float16 if sel_device == "cuda" and torch.cuda.is_available() and dtype_pref == "fp16" else torch.float32

    print(
        f"[selection] Starting PCL keyword scan for {llm_type} on {task}/{dataset} "
        f"(samples={n_samples}, k_shot={k_shot}, max_len={max_length})"
    )

    texts, labels = resolve_dataset(task, dataset, n_samples=n_samples, seed=seed, split="validation", stratified=True)
    if len(texts) < 8:
        cfg = AutoConfig.from_pretrained(llm_id)
        fallback = max(0, cfg.num_hidden_layers // 2)
        print("[selection] Not enough samples for selection; using middle layer", fallback)
        return fallback

    abstract = identify_abstract_layer(
        llm_id=llm_id,
        texts=texts,
        labels=labels,
        n_pcs=n_pcs,
        top_pc=top_pc,
        k_shot=k_shot,
        max_length=max_length,
        device=sel_device,
        dtype=dtype,
        keyword_top_k=keyword_top_k,
        keyword_source=keyword_source,
        keyword_llm_id=keyword_llm_id,
        keyword_sample_limit=keyword_sample_limit,
        keyword_weight=keyword_weight,
        seed=seed,
    )
    if abstract.layer < 0:
        cfg = AutoConfig.from_pretrained(llm_id)
        fallback = max(0, cfg.num_hidden_layers // 2)
        print("[selection] PCL scoring failed; using middle layer", fallback)
        return fallback

    print(
        f"[selection][Stage1] L_Abstract={abstract.layer} | "
        f"keyword_peak={abstract.per_layer_signals[abstract.layer]['keyword_signal']:.3f} "
        f"corr={abstract.per_layer_signals[abstract.layer]['corr_signal']:.3f} "
        f"(source={keyword_source}, top_k={keyword_top_k})"
    )
    if abstract.keywords:
        preview = abstract.keywords[: min(8, len(abstract.keywords))]
        print(f"[selection][Stage1] Keywords used: {preview}")

    apply = identify_apply_layer(
        llm_id=llm_id,
        abstract=abstract,
        lambda_scale=lambda_scale,
        max_length=max_length,
        batch_size=patch_batch_size,
        seed=seed,
        val_limit=patch_val_limit if patch_val_limit > 0 else None,
        max_layers=int(getattr(args, "selection_max_layers", 0)),
        max_heads=int(getattr(args, "selection_max_heads", 0)),
    )

    if apply.layer < 0 or apply.head_effects.size == 0 or float(np.max(apply.head_effects)) == 0.0:
        final_layer = abstract.layer
        print("[selection][Stage2] No strong causal heads detected; falling back to L_Abstract")
    else:
        final_layer = apply.layer
        print(
            f"[selection][Stage2] L_Apply={apply.layer} | "
            f"base_acc={apply.base_acc:.3f} | max_head_effect={apply.head_effects.max():.3f}"
        )

    # Optional JSON logging of selection signals for later analysis
    log_dir = getattr(args, "selection_log_dir", None)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir,
                f"{task}_{dataset}_{llm_type}_seed{seed}.json",
            )
            payload = {
                "task": task,
                "dataset": dataset,
                "llm_type": llm_type,
                "seed": seed,
                "n_samples": n_samples,
                "L_Abstract": int(abstract.layer),
                "L_Apply": int(apply.layer) if apply.layer is not None else -1,
                "final_layer": int(final_layer),
                "keyword_source": keyword_source,
                "keyword_top_k": keyword_top_k,
                "keywords": abstract.keywords,
                "per_layer_signals": [
                    {
                        "layer": int(i),
                        "keyword_signal": float(sig.get("keyword_signal", 0.0)),
                        "corr_signal": float(sig.get("corr_signal", 0.0)),
                    }
                    for i, sig in enumerate(abstract.per_layer_signals)
                ],
                "base_acc": float(apply.base_acc) if apply.layer >= 0 else None,
                "max_head_effect": float(apply.head_effects.max()) if apply.head_effects.size > 0 else None,
                # Full head_effects matrix for ILM heatmap visualization
                "head_effects": apply.head_effects.tolist() if apply.head_effects.size > 0 else None,
                # Per-layer probe accuracy for probe vs test accuracy analysis
                "per_layer_probe_acc": apply.per_layer_probe_acc,
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[selection][Stage3] Logged selection details to {log_path}")
        except Exception as e:
            print(f"[selection][Stage3] Failed to log selection details: {e}")

    if multi_layer_span > 0:
        num_layers_total = len(abstract.hidden_per_layer)
        candidates = set()
        for base in [abstract.layer, apply.layer]:
            if base is None or base < 0:
                continue
            for d in range(-multi_layer_span, multi_layer_span + 1):
                cand = base + d
                if 0 <= cand < num_layers_total:
                    candidates.add(int(cand))
        if candidates:
            cand_sorted = sorted(candidates)
            print(f"[selection][Stage3] Multi-layer window suggestion for ablations: {cand_sorted}")

    print(f"[selection][Stage3] Final L_LLM={final_layer} (default to L_Apply else L_Abstract)")
    return int(final_layer)
