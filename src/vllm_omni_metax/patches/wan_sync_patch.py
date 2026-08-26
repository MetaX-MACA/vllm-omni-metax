# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Wan2.2 decode GPU-sync workaround.

The vllm-omni omni stage pipeline can read the VAE decode result before the
GPU work is finished, producing corrupted video on the MetaX stack: uniform
gray noise, missing color channels, or NaN frames (mainly with 81-frame /
21-latent-frame requests).  Forcing a device sync right after
``Wan22Pipeline.forward`` (i.e. after VAE decode) makes the output
deterministic and clean.

The patch is applied lazily: the omni platform plugin activates while
vllm-omni is still initialising, so importing
``vllm_omni.diffusion.models.wan2_2`` there would hit a circular import.
Instead we wait quietly until vllm-omni is importable (short poll, no log
noise) and only then swap in the wrapper.  The wrapper itself imports
``current_omni_platform`` at call time (true lazy import).
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_PATCHED = False
_MAX_RETRIES = 600  # 1s interval; service startup can take minutes
_RETRY_COUNT = 0


def apply_wan_sync_patch() -> None:
    global _PATCHED, _RETRY_COUNT
    if _PATCHED:
        return

    try:
        from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2
    except Exception:
        # vllm_omni is still initialising; retry shortly afterwards.
        if _RETRY_COUNT < _MAX_RETRIES:
            _RETRY_COUNT += 1
            threading.Timer(1.0, apply_wan_sync_patch).start()
        else:
            logger.error(
                "wan sync patch: vllm_omni never became ready after %d retries.",
                _MAX_RETRIES,
            )
        return

    original_forward = pipeline_wan2_2.Wan22Pipeline.forward

    def _forward_with_sync(self, req):
        from vllm_omni.platforms import current_omni_platform  # lazy import

        output = original_forward(self, req)
        current_omni_platform.synchronize()
        return output

    pipeline_wan2_2.Wan22Pipeline.forward = _forward_with_sync
    _PATCHED = True
    logger.info("Patched Wan22Pipeline.forward with post-decode GPU sync.")
