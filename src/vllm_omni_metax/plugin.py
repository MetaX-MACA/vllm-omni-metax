# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _detect_metax_with_mxsml() -> bool:
    """Return True if the current host looks like a usable MetaX runtime.

    We detect through `vllm_metax.utils.import_pymxsml()` instead of touching any
    `vllm-omni` builtin CUDA plugin. This keeps the ownership boundary clean:
    `vllm-metax` owns the hardware backend and this plugin only asks whether it
    exists.
    """
    try:
        from vllm_metax.utils import import_pymxsml

        pymxsml = import_pymxsml()
        pymxsml.nvmlInit()
        try:
            count = int(pymxsml.nvmlDeviceGetCount())
            if count <= 0:
                logger.debug("MetaX plugin probe found zero visible devices.")
                return False

            # Best-effort logging only; names are not required for activation.
            try:
                names = []
                for idx in range(count):
                    handle = pymxsml.nvmlDeviceGetHandleByIndex(idx)
                    names.append(str(pymxsml.nvmlDeviceGetName(handle)))
                logger.info("Detected MetaX devices via mxsml: %s", names)
            except Exception:
                logger.debug("MetaX device-name probe failed; continuing.", exc_info=True)

            return True
        finally:
            try:
                pymxsml.nvmlShutdown()
            except Exception:
                logger.debug("mxsml shutdown failed during plugin probe.", exc_info=True)
    except Exception:
        logger.debug("MetaX plugin probe failed.", exc_info=True)
        return False
    
def _apply_metax_patches() -> None:
    if _env_flag("VLLM_OMNI_METAX_DISABLE_PATCHES"):
        logger.info("vllm-omni-metax patches disabled by VLLM_OMNI_METAX_DISABLE_PATCHES.")
        return
    
    try:
        from vllm_omni_metax.patches import (
            apply_rope_patch,
            apply_metax_qwen3_tts_runtime_patches,
        )
        from vllm_omni_metax.models import register_metax_omni_models

        apply_rope_patch()
        register_metax_omni_models()
        apply_metax_qwen3_tts_runtime_patches()
    except Exception:
        logger.warning("Failed to apply vllm-omni-metax patches.", exc_info=True)

def metax_omni_platform_plugin() -> Optional[str]:
    """Entry point for `vllm_omni.platform_plugins`.

    Important behavior:
    - It does not modify builtin omni plugins.
    - It only returns a class path when MetaX runtime is available.
    - When returned, `vllm-omni` will prefer this out-of-tree plugin over its
      builtin CUDA plugin.
    """
    if _env_flag("VLLM_OMNI_METAX_DISABLE"):
        logger.info("vllm-omni-metax plugin disabled by VLLM_OMNI_METAX_DISABLE.")
        return None

    if _env_flag("VLLM_OMNI_METAX_FORCE"):
        logger.warning("vllm-omni-metax forced on by VLLM_OMNI_METAX_FORCE.")
        _apply_metax_patches()
        return "vllm_omni_metax.platform.MetaxOmniPlatform"

    if _detect_metax_with_mxsml():
        logger.info("Activating vllm-omni-metax platform plugin.")
        _apply_metax_patches()
        return "vllm_omni_metax.platform.MetaxOmniPlatform"

    return None

