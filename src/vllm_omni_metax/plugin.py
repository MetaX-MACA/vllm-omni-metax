# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import logging
import os
import threading
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

    logger.info("Applying vllm-omni-metax patches...")

    # Apply each patch independently — a failure in one must not block the
    # others.  deploy resolution runs first so subsequent patches can rely on
    # deploy config values provided by the plugin's YAML.

    # 1. Deploy resolution patch
    try:
        from vllm_omni_metax.patches.deploy_resolution_patch import apply_deploy_resolution_patch

        apply_deploy_resolution_patch()
    except Exception:
        logger.warning("Failed to apply deploy_resolution patch.", exc_info=True)

    # 2. Code predictor dtype fix (Qwen3-Omni talker forward)
    try:
        from vllm_omni_metax.patches import apply_code_predictor_patch

        apply_code_predictor_patch()
    except Exception:
        logger.warning("Failed to apply code_predictor patch.", exc_info=True)

    # 3. RoPE fallback shim (vllm.vllm_flash_attn.layers.rotary)
    try:
        from vllm_omni_metax.patches import apply_rope_patch

        apply_rope_patch()
    except Exception:
        logger.warning("Failed to apply rope patch.", exc_info=True)

    # 4. Qwen3-TTS runtime patches (Code2Wav cudagraph / Triton SnakeBeta)
    try:
        from vllm_omni_metax.patches import apply_metax_qwen3_tts_runtime_patches

        apply_metax_qwen3_tts_runtime_patches()
    except Exception:
        logger.warning("Failed to apply qwen3-tts runtime patch.", exc_info=True)

    # Deploy resolution patch may have been deferred because
    # config_factory was still initialising during platform detection.
    # Retry after a short delay so the import call stack can unwind.
    def _retry_deploy():
        try:
            from vllm_omni_metax.patches.deploy_resolution_patch import ensure_patch_installed

            ensure_patch_installed()
        except Exception:
            pass

    threading.Timer(0.5, _retry_deploy).start()

    logger.info("vllm-omni-metax patches applied.")


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
