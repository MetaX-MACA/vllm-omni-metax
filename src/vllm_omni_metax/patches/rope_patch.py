# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
import types

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _torch_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    conjugate: bool = False,
    **kwargs,
) -> torch.Tensor:
    # Basic torch fallback for flash_attn.layers.rotary.apply_rotary_emb.
    # Supports common qwen-image path shapes:
    #   [batch, seq, heads, dim]
    #   [tokens, heads, dim]
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)

    if conjugate:
        sin = -sin

    rotary_dim = cos.shape[-1] * 2
    x_ro = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    if interleaved:
        cos_full = torch.repeat_interleave(cos, 2, dim=-1)
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)
    else:
        cos_full = torch.cat((cos, cos), dim=-1)
        sin_full = torch.cat((sin, sin), dim=-1)

    # Broadcast cos/sin to x_ro.
    if x_ro.dim() == 4:
        # [B, S, H, D]
        seq_len = x_ro.shape[1]
        cos_full = cos_full[:seq_len].unsqueeze(0).unsqueeze(2)
        sin_full = sin_full[:seq_len].unsqueeze(0).unsqueeze(2)
    elif x_ro.dim() == 3:
        # [T, H, D]
        seq_len = x_ro.shape[0]
        cos_full = cos_full[:seq_len].unsqueeze(1)
        sin_full = sin_full[:seq_len].unsqueeze(1)
    else:
        # Fallback: append singleton dims before last dim.
        while cos_full.dim() < x_ro.dim():
            cos_full = cos_full.unsqueeze(0)
            sin_full = sin_full.unsqueeze(0)

    out_ro = x_ro * cos_full + _rotate_half(x_ro, interleaved) * sin_full

    if x_pass.numel() == 0:
        out = out_ro
    else:
        out = torch.cat((out_ro, x_pass), dim=-1)

    if inplace:
        x.copy_(out)
        return x

    return out

_USE_EXT_ROTARY = None

def _metax_apply_rotary_emb(*args, **kwargs):
    global _USE_EXT_ROTARY
    if _USE_EXT_ROTARY is None:
        try:
            from flash_attn.layers.rotary import apply_rotary_emb as ext_fn
            ext_fn(*args, **kwargs)  # test call
            _USE_EXT_ROTARY = True
        except Exception:
            _USE_EXT_ROTARY = False

    if _USE_EXT_ROTARY:
        return ext_apply_rotary_emb(*args, **kwargs)
    return _torch_apply_rotary_emb(*args, **kwargs)

def _install_vllm_flash_attn_rotary_shim() -> None:
    """Make vllm.vllm_flash_attn.layers.rotary importable on MetaX.

    This prevents vllm-omni rope.py from importing the CUDA-only vLLM FA ext.
    """
    vllm_fa_mod = types.ModuleType("vllm.vllm_flash_attn")
    vllm_fa_mod.__path__ = []

    layers_mod = types.ModuleType("vllm.vllm_flash_attn.layers")
    layers_mod.__path__ = []

    rotary_mod = types.ModuleType("vllm.vllm_flash_attn.layers.rotary")
    rotary_mod.apply_rotary_emb = _metax_apply_rotary_emb

    sys.modules["vllm.vllm_flash_attn"] = vllm_fa_mod
    sys.modules["vllm.vllm_flash_attn.layers"] = layers_mod
    sys.modules["vllm.vllm_flash_attn.layers.rotary"] = rotary_mod


def apply_rope_patch() -> None:
    global _PATCHED

    if _PATCHED:
        return

    _install_vllm_flash_attn_rotary_shim()

    # Optional: also patch already-imported RotaryEmbedding.forward_cuda path.
    try:
        from vllm_omni.diffusion.layers import rope

        if hasattr(rope, "RotaryEmbedding"):
            cls = rope.RotaryEmbedding

            if hasattr(cls, "forward_cuda"):
                original_forward_cuda = cls.forward_cuda

                def patched_forward_cuda(self, *args, **kwargs):
                    return original_forward_cuda(self, *args, **kwargs)

                cls.forward_cuda = patched_forward_cuda
                logger.info("Patched vllm-omni RotaryEmbedding.forward_cuda.")
    except Exception:
        logger.debug(
            "RotaryEmbedding class patch skipped; shim is already installed.",
            exc_info=True,
        )

    _PATCHED = True
    logger.info("Installed MetaX vllm_flash_attn rotary shim.")