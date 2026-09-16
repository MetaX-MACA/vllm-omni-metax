# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Force the Qwen3-Omni-MoE thinker-only pipeline (no talker / code2wav).

vllm-omni picks the Qwen3-Omni pipeline variant from the HF config flag
``enable_audio_output``: ``resolve_qwen3_omni_pipeline`` returns the
single-stage ``QWEN3_OMNI_THINKER_ONLY_PIPELINE`` when the flag is false and
the 3-stage thinker+talker+code2wav ``QWEN3_OMNI_PIPELINE`` otherwise.
There is no deploy-YAML or CLI switch for it (``pipeline:`` in a deploy YAML
only accepts registered ``OMNI_PIPELINES`` keys, and the registry entry for
``qwen3_omni_moe`` *is* that resolver), so text-only MetaX deployments would
have to edit the served checkpoint's ``config.json`` -- often impossible when
the checkpoint lives on a read-only share.

This patch swaps the resolver, so the thinker-only variant is selected
regardless of the checkpoint flag:

* module attribute ``vllm_omni.model_executor.models.qwen3_omni.pipeline.resolve_qwen3_omni_pipeline``
* registry entry ``vllm_omni.config.pipeline_registry.OMNI_PIPELINES["qwen3_omni_moe"]``
  (the registry captured the original callable at import time)

It is opt-in via ``VLLM_OMNI_METAX_QWEN3_OMNI_THINKER_ONLY=1`` so the shared
default stays the full thinker+talker+code2wav pipeline.

Effects: only the thinker weights are loaded (2 GPUs on C500 instead of 3),
responses are text only (no ``/v1/audio/speech`` / audio track), and any
talker / code2wav entries of a deploy YAML are ignored -- ``merge_pipeline_deploy``
walks the pipeline's stage list.  Stage 0 keeps the deployment knobs from the
deploy YAML (``qwen3_omni_moe.yaml`` stage 0: TP=2 on devices "0,1",
gpu_memory_utilization 0.9) or from
``vllm_omni_metax/deploy/qwen3_omni_moe_thinker.yaml`` when passed explicitly.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_PATCHED = False
_RETRY_COUNT = 0
_MAX_RETRIES = 600  # 1s interval; the plugin may activate while vllm-omni loads


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def apply_qwen3_omni_thinker_only_patch() -> None:
    """Select the Qwen3-Omni-MoE thinker-only pipeline unconditionally.

    Purpose: run Qwen3-Omni text-only (thinker stage) without editing the
    served checkpoint's ``enable_audio_output`` flag, which is read-only on the
    MetaX registry shares.  Activate with
    ``VLLM_OMNI_METAX_QWEN3_OMNI_THINKER_ONLY=1``.

    Added: V0.26.0.
    remove_at: upstream exposes a supported switch for the thinker-only variant
    (CLI flag or deploy-YAML key) instead of relying on the checkpoint's
    ``enable_audio_output`` flag.
    """
    global _PATCHED, _RETRY_COUNT

    if _PATCHED:
        return

    if not _env_flag("VLLM_OMNI_METAX_QWEN3_OMNI_THINKER_ONLY"):
        logger.debug(
            "MetaX: Qwen3-Omni thinker-only patch not requested "
            "(VLLM_OMNI_METAX_QWEN3_OMNI_THINKER_ONLY != 1)."
        )
        return

    try:
        from vllm_omni.config import pipeline_registry
        from vllm_omni.model_executor.models.qwen3_omni import pipeline as qwen3_omni_pipeline
    except Exception:
        # vllm-omni is still initialising; retry shortly afterwards.
        if _RETRY_COUNT < _MAX_RETRIES:
            _RETRY_COUNT += 1
            threading.Timer(1.0, apply_qwen3_omni_thinker_only_patch).start()
        else:
            logger.error(
                "MetaX: Qwen3-Omni thinker-only patch could not import vllm-omni after %d retries.",
                _MAX_RETRIES,
            )
        return

    thinker_only_pipeline = qwen3_omni_pipeline.QWEN3_OMNI_THINKER_ONLY_PIPELINE

    def _resolver(hf_config=None):
        # Signature-compatible with resolve_qwen3_omni_pipeline(hf_config).
        return thinker_only_pipeline

    qwen3_omni_pipeline.resolve_qwen3_omni_pipeline = _resolver

    try:
        pipeline_registry.OMNI_PIPELINES["qwen3_omni_moe"] = _resolver
    except Exception:
        logger.error(
            "MetaX: could not update OMNI_PIPELINES['qwen3_omni_moe']; "
            "the thinker-only pipeline may not be selected.",
            exc_info=True,
        )

    _PATCHED = True
    logger.warning(
        "MetaX: Qwen3-Omni forced to the thinker-only pipeline "
        "(VLLM_OMNI_METAX_QWEN3_OMNI_THINKER_ONLY=1): stage 0 only, text output, "
        "talker/code2wav will not be started."
    )
