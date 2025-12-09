from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def pool_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Pool token representations into a single vector per sequence."""
    if hidden.size(1) > 1:
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        return hidden[:, 0]
    return hidden[:, 0]


@torch.no_grad()
def get_llm_hidden_layers(
    llm_id: str,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 128,
    batch_size: int = 16,
    dtype: Optional[torch.dtype] = None,
    pooling: str = "first",
    l2_norm: bool = True,
) -> List[np.ndarray]:
    """Encode texts with an LLM and return pooled hidden states per layer."""
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

    enc0 = tokenizer(texts[:1], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc0 = {k: v.to(device) for k, v in enc0.items()}
    out0 = model(**enc0, output_hidden_states=True)
    n_layers_total = len(out0.hidden_states) - 1
    all_layers: List[np.ndarray] = [
        np.empty((0, out0.hidden_states[-1].size(-1)), dtype=np.float32) for _ in range(n_layers_total)
    ]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = list(out.hidden_states)[1:]
        for li, h in enumerate(hs):
            if pooling == "first":
                pooled = h[:, 0]
            else:
                pooled = pool_hidden(h, enc.get("attention_mask"))
            pooled_np = pooled.detach().float().cpu().numpy()
            if l2_norm:
                norm = np.linalg.norm(pooled_np, axis=1, keepdims=True)
                pooled_np = pooled_np / np.maximum(norm, 1e-8)
            all_layers[li] = np.vstack([all_layers[li], pooled_np])

    return all_layers
