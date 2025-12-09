from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _find_subsequence(seq: List[int], pattern: List[int]) -> Optional[int]:
    """Return start index of pattern in seq or None."""
    if len(pattern) == 0 or len(seq) == 0 or len(pattern) > len(seq):
        return None
    for i in range(len(seq) - len(pattern) + 1):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return None


@torch.no_grad()
def get_label_token_hidden_per_layer(
    llm_id: str,
    prompts: List[str],
    label_token_strs: List[List[str]],
    device: str = "cuda",
    max_length: int = 256,
    batch_size: int = 8,
    dtype: Optional[torch.dtype] = None,
    return_label_embs: bool = False,
    fallback: str = "none",
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """
    Encode prompts and collect per-layer hidden states at label-token positions.

    Args:
        llm_id: HF model id
        prompts: list of prompt strings
        label_token_strs: list of token strings per example (usually length-1)
        device: cuda/cpu
        max_length: truncation length
        batch_size: batch size
        dtype: torch dtype (defaults to fp16 on cuda)

    Returns:
        List[np.ndarray] per layer, stacked over examples (N, hidden_dim)
    """
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

    # prime for num_layers and dim
    enc0 = tokenizer(prompts[:1], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc0 = {k: v.to(device) for k, v in enc0.items()}
    out0 = model(**enc0, output_hidden_states=True)
    n_layers_total = len(out0.hidden_states) - 1
    hidden_dim = out0.hidden_states[-1].size(-1)
    per_layer: List[List[np.ndarray]] = [[] for _ in range(n_layers_total)]

    # Pre-tokenize label tokens
    label_token_ids: List[List[int]] = []
    for toks in label_token_strs:
        # join to allow multi-token labels if present
        tok_ids = tokenizer(" ".join(toks), add_special_tokens=False)["input_ids"]
        label_token_ids.append(tok_ids)
    label_embs: List[np.ndarray] = []
    match_success = 0
    match_fail = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_lab_ids = label_token_ids[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = list(out.hidden_states)[1:]  # drop embedding layer
        input_ids = enc["input_ids"].tolist()

        for b_idx, ids in enumerate(input_ids):
            lab_ids = batch_lab_ids[b_idx] if b_idx < len(batch_lab_ids) else []
            start = _find_subsequence(ids, lab_ids) if lab_ids else None
            if start is None:
                if fallback == "cls":
                    start, end = 0, 1
                elif fallback == "last":
                    start = len(ids) - 2 if len(ids) > 1 else 0
                    end = start + 1
                else:
                    start = len(ids) - 2 if len(ids) > 1 else 0
                    end = start + 1
                match_fail += 1
            else:
                end = min(start + len(lab_ids), len(ids))
                match_success += 1
            if return_label_embs:
                # Average embedding of label token ids for this example
                if lab_ids:
                    emb_vecs = model.get_input_embeddings().weight[lab_ids, :]
                    lab_emb = emb_vecs.mean(dim=0)
                else:
                    lab_emb = model.get_input_embeddings().weight[ids[-1], :]
                label_embs.append(lab_emb.detach().float().cpu().numpy())
            for li, h in enumerate(hs):
                token_span = h[b_idx, start:end, :]  # (span, dim)
                pooled = token_span.mean(dim=0)
                per_layer[li].append(pooled.detach().float().cpu().numpy())

    hidden_per_layer = [
        np.stack(layer_list, axis=0) if len(layer_list) > 0 else np.empty((0, hidden_dim), dtype=np.float32)
        for layer_list in per_layer
    ]
    if return_label_embs:
        lab_arr = np.stack(label_embs, axis=0) if len(label_embs) > 0 else np.empty((0, hidden_dim), dtype=np.float32)
        return hidden_per_layer, lab_arr, {"match_success": match_success, "match_fail": match_fail}
    return hidden_per_layer, None, {"match_success": match_success, "match_fail": match_fail}
