# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Clamp SDPA additive masks that saturate the bfloat16 range.

On MetaX (torch 2.10.0+metax3.8.2.2, MACA 3.8.2.x)
``torch.nn.functional.scaled_dot_product_attention`` returns an all-NaN tensor
when all of the following hold at once:

* query/key/value are bfloat16,
* the additive (float) ``attn_mask`` contains values around
  ``torch.finfo(torch.bfloat16).min`` (``-3.39e38``; ``-2e38``, ``-1e4`` and
  ``-inf`` are handled correctly),
* the explicit ``scale`` is ``>= 1.0`` (``<= 0.5`` is fine), and
* the sequence length is ``>= 128`` (``<= 64`` is fine).

All three SDPA backends (flash / mem_efficient / math) are affected, while the
mathematically identical ``matmul + softmax`` (eager attention) path is not.

HuggingFace's SDPA dispatch builds exactly such a mask: with a boolean
attention mask, ``transformers.integrations.sdpa_attention.create_position_bias_mask``
fills the masked positions with ``torch.finfo(key.dtype).min``.  UMT5 -- the
Wan2.1/Wan2.2 text encoder -- additionally passes ``scale=1.0``
(``UMT5Attention.__init__`` sets ``self.scaling = 1.0``) and pads prompts to
512 tokens, so the whole text encoder output becomes NaN, every denoise step
stays NaN and Wan2.2 T2V answers HTTP 200 with a 49-frame all-black clip
(``np.round(frames).astype(np.uint8)`` turns NaN into 0).

Clamping the mask to a large but safe negative value (default ``-1e4``) keeps
the softmax semantics exact -- ``exp(x)`` is 0 for every masked entry -- while
staying inside the range the MetaX kernels handle.  The clamp only runs when
the effective scale is ``>= 1.0`` (the single failing configuration), so
ordinary LLM decode paths (``scale = head_dim ** -0.5 < 1``) pay nothing.

Env controls:

* ``VLLM_OMNI_METAX_DISABLE_SDPA_MASK_CLAMP=1`` -- skip this patch entirely.
* ``VLLM_OMNI_METAX_SDPA_MASK_MIN`` -- override the clamp floor (default -1e4).
* ``VLLM_OMNI_METAX_SDPA_MASK_CLAMP_ALL=1`` -- clamp regardless of ``scale``.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_PATCHED = False
_RETRY_COUNT = 0
_MAX_RETRIES = 600  # 1s interval; the plugin may activate while transformers loads

# -1e4 keeps `exp(mask)` at exactly 0 for masked positions in bf16/fp16 while
# remaining far away from the values that trigger the MetaX SDPA NaN.
_DEFAULT_MASK_MIN = -1e4


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _mask_min() -> float:
    raw = os.getenv("VLLM_OMNI_METAX_SDPA_MASK_MIN", "").strip()
    if not raw:
        return _DEFAULT_MASK_MIN
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "MetaX: ignoring invalid VLLM_OMNI_METAX_SDPA_MASK_MIN=%r; using %s.",
            raw,
            _DEFAULT_MASK_MIN,
        )
        return _DEFAULT_MASK_MIN


def apply_sdpa_min_mask_patch() -> None:
    """Clamp additive SDPA masks that use ``torch.finfo(dtype).min``.

    Purpose: prevent the MetaX SDPA NaN described in the module docstring.
    The mask is clamped at the two places HuggingFace's SDPA dispatch can pick
    it up: the ``create_position_bias_mask`` helper (boolean-mask models such
    as UMT5) and the ``sdpa_attention_forward`` entry point itself (models
    that pass an already additive float mask).  Registry entries that captured
    the original function at import time are updated as well.

    Added: V0.26.0.
    remove_at: MetaX torch/MACA SDPA computes correctly for additive masks
    containing ``torch.finfo(dtype).min`` (repro: bf16 q/k/v + ``finfo.min``
    mask + ``scale>=1.0`` + seq>=128, verified NaN on flash/mem_efficient/math
    with torch 2.10.0+metax3.8.2.2).
    """
    global _PATCHED, _RETRY_COUNT

    if _PATCHED:
        return

    if _env_flag("VLLM_OMNI_METAX_DISABLE_SDPA_MASK_CLAMP"):
        logger.warning(
            "MetaX: SDPA mask clamp disabled by VLLM_OMNI_METAX_DISABLE_SDPA_MASK_CLAMP."
        )
        return

    try:
        import torch
    except Exception:
        logger.debug("MetaX: SDPA mask clamp skipped (torch unavailable).", exc_info=True)
        return

    try:
        from transformers.integrations import sdpa_attention
    except Exception:
        # transformers is still initialising; retry shortly afterwards.
        if _RETRY_COUNT < _MAX_RETRIES:
            _RETRY_COUNT += 1
            threading.Timer(1.0, apply_sdpa_min_mask_patch).start()
        else:
            logger.error(
                "MetaX: SDPA mask clamp could not import transformers after %d retries.",
                _MAX_RETRIES,
            )
        return

    mask_min = _mask_min()
    clamp_all = _env_flag("VLLM_OMNI_METAX_SDPA_MASK_CLAMP_ALL")

    def _clamp(mask):
        if isinstance(mask, torch.Tensor) and mask.is_floating_point():
            return torch.clamp(mask, min=mask_min)
        return mask

    original_create_position_bias_mask = getattr(
        sdpa_attention, "create_position_bias_mask", None
    )
    original_sdpa_attention_forward = sdpa_attention.sdpa_attention_forward

    if original_create_position_bias_mask is not None and not getattr(
        original_create_position_bias_mask, "_metax_mask_clamped", False
    ):

        def create_position_bias_mask(*args, **kwargs):
            return _clamp(original_create_position_bias_mask(*args, **kwargs))

        create_position_bias_mask._metax_mask_clamped = True
        sdpa_attention.create_position_bias_mask = create_position_bias_mask

    def sdpa_attention_forward(*args, **kwargs):
        # (module, query, key, value, attention_mask, ...); support both call styles.
        query = args[1] if len(args) > 1 else kwargs.get("query")
        scale = args[5] if len(args) > 5 else kwargs.get("scaling")
        if scale is None and torch.is_tensor(query):
            scale = query.shape[-1] ** -0.5
        # Below 1.0 the MetaX kernels handle finfo.min masks correctly, so the
        # clamp is skipped and the hot decode path keeps its original cost.
        if clamp_all or (scale is not None and scale >= 1.0):
            if len(args) > 4:
                args = args[:4] + (_clamp(args[4]),) + args[5:]
            elif "attention_mask" in kwargs:
                kwargs["attention_mask"] = _clamp(kwargs["attention_mask"])
        return original_sdpa_attention_forward(*args, **kwargs)

    sdpa_attention.sdpa_attention_forward = sdpa_attention_forward

    # Entry points that captured the original callables at import time.
    try:
        from transformers import modeling_utils

        registry = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
        if registry is not None and registry.get("sdpa") is not sdpa_attention_forward:
            registry["sdpa"] = sdpa_attention_forward
    except Exception:
        logger.debug("MetaX: SDPA mask clamp could not update the registry.", exc_info=True)

    try:
        import sys

        msa_attention = sys.modules.get("transformers.integrations.msa_attention")
        if msa_attention is not None and (
            getattr(msa_attention, "sdpa_attention_forward", None)
            is original_sdpa_attention_forward
        ):
            msa_attention.sdpa_attention_forward = sdpa_attention_forward
    except Exception:
        logger.debug("MetaX: SDPA mask clamp could not patch msa_attention.", exc_info=True)

    _PATCHED = True
    logger.info(
        "MetaX: clamped SDPA additive masks to >= %s (scale>=1.0%s) to avoid NaN output.",
        mask_min,
        " or always" if clamp_all else "",
    )
