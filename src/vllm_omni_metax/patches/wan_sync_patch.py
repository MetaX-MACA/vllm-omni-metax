# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Wan2.2 decode GPU-sync workaround.

The vllm-omni omni stage pipeline can read the VAE decode result before the
GPU work is finished, producing corrupted video on the MetaX stack: uniform
gray noise, missing color channels, or NaN frames (mainly with 81-frame /
21-latent-frame requests).  Forcing a device sync right after
``Wan22Pipeline.forward`` (i.e. after VAE decode) makes the output
deterministic and clean.
"""
from __future__ import annotations

import logging

from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_wan_sync_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2

    original_forward = pipeline_wan2_2.Wan22Pipeline.forward

    def _forward_with_sync(self, req):
        output = original_forward(self, req)
        current_omni_platform.synchronize()
        return output

    pipeline_wan2_2.Wan22Pipeline.forward = _forward_with_sync
    _PATCHED = True
    logger.info("Patched Wan22Pipeline.forward with post-decode GPU sync.")
