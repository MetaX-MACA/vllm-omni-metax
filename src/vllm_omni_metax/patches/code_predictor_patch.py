# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _patched_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    from vllm_omni.platforms import current_omni_platform

    bsz, seq_len, _ = hidden_states.shape
    hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
    hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

    qkv = self.qkv_proj(hidden_states)
    q_raw, k_raw, v_raw = self._split_qkv(qkv)
    q = self.q_norm(q_raw.view(hidden_shape_q)).transpose(1, 2)
    k = self.k_norm(k_raw.view(hidden_shape_kv)).transpose(1, 2)
    v = v_raw.view(hidden_shape_kv).transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    if current_omni_platform.is_npu():
        import torch_npu

        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)
        attn_out = self._forward_npu_attention(q, k, v, bsz, seq_len)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self.is_gqa,
        )

    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    # /------------------------  Metax Modification -------------------------\
    attn_out = attn_out.to(self.o_proj.weight.dtype)  # MetaX: ensure dtype match
    # \------------------------  Metax Modification -------------------------/
    return self.o_proj(attn_out)


def apply_code_predictor_patch() -> None:
    """Monkey-patch CodePredictorAttention.forward for MetaX dtype compatibility."""
    global _PATCHED

    if _PATCHED:
        return

    try:
        from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
            CodePredictorAttention,
        )

    except ImportError:
        logger.debug("CodePredictorAttention not available; skipping dtype patch.")
        return

    CodePredictorAttention.forward = _patched_attention_forward
    _PATCHED = True
    logger.info("Patched CodePredictorAttention.forward (dtype cast before o_proj).")