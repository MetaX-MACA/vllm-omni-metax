# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_metax_omni_models() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    metax_module = "vllm_omni_metax.models.qwen3_tts_code2wav"
    metax_class = "Qwen3TTSCode2Wav"

    # vLLM-Omni v0.20.0: use OmniModelRegistry, not ModelRegistry.
    OmniModelRegistry.register_model(
        "Qwen3TTSCode2Wav",
        f"{metax_module}:{metax_class}",
    )

    logger.warning(
        "MetaX: patched OmniModelRegistry Qwen3TTSCode2Wav -> %s:%s",
        metax_module,
        metax_class,
    )

    _REGISTERED = True
