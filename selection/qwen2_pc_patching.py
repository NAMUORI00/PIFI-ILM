from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention


@dataclass
class HeadPCs:
    """Store ILM PCs for a single head."""

    pcs: np.ndarray  # (k, head_dim)


@dataclass
class ILMHeadPatchState:
    """
    Global state used by the patched Qwen2SdpaAttention.forward.

    This keeps the implementation simple and avoids modifying the Qwen2 model
    classes directly. The patched forward checks this state on every call.
    """

    # Collection mode: accumulate per-head representations for PCA.
    collect: bool = False
    # Mapping (layer_idx -> list of head vectors per example)
    # Each entry is a list of np.ndarray of shape (num_heads, head_dim).
    collected: Dict[int, List[np.ndarray]] = field(default_factory=dict)

    # Patching mode: apply ILM PC patching to a specific (layer, head).
    enabled: bool = False
    target_layer: Optional[int] = None
    target_head: Optional[int] = None
    # PCs for the current (layer, head), shape (k, head_dim) in torch tensor.
    pcs_torch: Optional[torch.Tensor] = None
    # Patch strength: z' = z - lambda_scale * proj(z)
    lambda_scale: float = 1.0


STATE = ILMHeadPatchState()


def reset_collection() -> None:
    """Reset collection buffers."""
    STATE.collect = False
    STATE.collected.clear()


def start_collection() -> None:
    """Enable collection of per-head representations."""
    reset_collection()
    STATE.collect = True


def stop_collection() -> None:
    """Disable collection (keeps buffers for PCA)."""
    STATE.collect = False


def enable_patching(
    layer_idx: int,
    head_idx: int,
    pcs: np.ndarray,
    device: torch.device,
    lambda_scale: float = 1.0,
) -> None:
    """Enable ILM PC patching for a specific (layer, head)."""
    if pcs.ndim == 1:
        pcs = pcs[None, :]
    pcs_t = torch.as_tensor(pcs, dtype=torch.float32, device=device)
    STATE.enabled = True
    STATE.target_layer = int(layer_idx)
    STATE.target_head = int(head_idx)
    STATE.pcs_torch = pcs_t
    STATE.lambda_scale = float(lambda_scale)


def disable_patching() -> None:
    """Disable any active patching."""
    STATE.enabled = False
    STATE.target_layer = None
    STATE.target_head = None
    STATE.pcs_torch = None


def _collect_head_vectors(layer_idx: int, attn_output: torch.Tensor) -> None:
    """
    Collect per-head vectors for PCA.

    Args:
        layer_idx: Index of the current layer.
        attn_output: Tensor of shape (batch, num_heads, seq_len, head_dim).
    """
    if not STATE.collect:
        return

    # Mean over sequence length for stability -> (batch, num_heads, head_dim)
    head_vecs = attn_output.mean(dim=2)  # (B, H, D)
    if layer_idx == 0 and not STATE.collected:
        print(f"[pc_patching] Collecting head vectors: shape={tuple(head_vecs.shape)}")
    head_vecs_np = head_vecs.detach().float().cpu().numpy()

    bucket = STATE.collected.setdefault(layer_idx, [])
    for i in range(head_vecs_np.shape[0]):
        bucket.append(head_vecs_np[i])  # (num_heads, head_dim)


def _apply_pc_patching(layer_idx: int, attn_output: torch.Tensor) -> torch.Tensor:
    """
    Apply ILM PC patching on the specified head (if active).

    Args:
        layer_idx: Index of the current layer.
        attn_output: Tensor of shape (batch, num_heads, seq_len, head_dim).

    Returns:
        Patched attn_output with same shape.
    """
    if not STATE.enabled:
        return attn_output
    if STATE.target_layer is None or STATE.target_head is None:
        return attn_output
    if layer_idx != STATE.target_layer:
        return attn_output
    if STATE.pcs_torch is None:
        return attn_output

    # attn_output: (B, H, T, D)
    B, H, T, D = attn_output.shape
    h_idx = STATE.target_head
    if not (0 <= h_idx < H):
        return attn_output

    pcs = STATE.pcs_torch  # (k, D)
    if pcs.numel() == 0:
        return attn_output

    # Flatten over (B, T) to apply patch efficiently.
    head_out = attn_output[:, h_idx, :, :]  # (B, T, D)
    flat = head_out.reshape(B * T, D)  # (N, D)
    orig_dtype = flat.dtype

    # Project onto ILM PCs and subtract projection: z' = z - sum_k (z·u_k) u_k
    proj_coeffs = flat.to(torch.float32) @ pcs.t()  # (N, k)
    proj_vecs = proj_coeffs @ pcs  # (N, D)
    flat_patched = flat.to(torch.float32) - STATE.lambda_scale * proj_vecs
    flat_patched = flat_patched.to(orig_dtype)

    head_out_patched = flat_patched.view(B, T, D)
    attn_output[:, h_idx, :, :] = head_out_patched
    return attn_output


_ORIGINAL_FORWARD = Qwen2SdpaAttention.forward


def patched_qwen2_sdpa_forward(
    self: Qwen2SdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    Drop-in replacement for Qwen2SdpaAttention.forward with ILM head collection/patching.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # (B, H, T, D)
    attn_output = self._scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        is_causal=self.is_causal,
        dropout=self.attention_dropout if self.training else 0.0,
        is_cross_attention=self.is_cross_attention,
    )

    _collect_head_vectors(self.layer_idx, attn_output)
    attn_output = _apply_pc_patching(self.layer_idx, attn_output)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    else:
        attn_weights = None  # not collected

    if use_cache:
        next_cache = ((key_states, value_states),)
    else:
        next_cache = None

    return attn_output, attn_weights, next_cache


class Qwen2HeadPatchContext:
    """
    Context manager that patches Qwen2SdpaAttention.forward to enable
    head-level PC collection/patching during ILM analysis.
    """

    def __enter__(self):
        Qwen2SdpaAttention.forward = patched_qwen2_sdpa_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Qwen2SdpaAttention.forward = _ORIGINAL_FORWARD
        disable_patching()
        reset_collection()

