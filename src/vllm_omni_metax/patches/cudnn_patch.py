# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Optional torch.backends.cudnn toggle.

Wan2.2 VAE decoding (and other diffusion convs) can hit
MCDNN_STATUS_INVALID_VALUE on MetaX.  Falling back to torch's native conv
implementations (cudnn disabled) makes them work correctly.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def apply_cudnn_patch() -> None:
    if _env_flag("VLLM_OMNI_METAX_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False
        logger.info(
            "vllm-omni-metax: torch.backends.cudnn.enabled=False "
            "(MCDNN VAE conv workaround)."
        )
    else:
        logger.debug("vllm-omni-metax: cudnn toggle not requested.")
