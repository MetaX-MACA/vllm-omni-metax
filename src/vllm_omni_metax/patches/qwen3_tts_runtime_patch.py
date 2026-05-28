# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _patch_code2wav_cudagraph() -> None:
    if os.getenv("VLLM_OMNI_METAX_ENABLE_CODE2WAV_CUDAGRAPH", "0") == "1":
        logger.warning(
            "MetaX: Qwen3-TTS Code2Wav CUDA Graph is explicitly enabled."
        )
        return

    try:
        from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
            CUDAGraphDecoderWrapper,
        )
        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Decoder,
        )
    except Exception:
        logger.debug("MetaX: Code2Wav CUDA Graph patch skipped.", exc_info=True)
        return

    if not getattr(Qwen3TTSTokenizerV2Decoder, "_metax_cudagraph_patched", False):

        def _skip_enable_cudagraph(self, *args, **kwargs):
            logger.warning(
                "MetaX: disabled Qwen3-TTS Code2Wav CUDA Graph; "
                "set VLLM_OMNI_METAX_ENABLE_CODE2WAV_CUDAGRAPH=1 to enable."
            )
            return None

        Qwen3TTSTokenizerV2Decoder.enable_cudagraph = _skip_enable_cudagraph
        Qwen3TTSTokenizerV2Decoder._metax_cudagraph_patched = True
        logger.warning("MetaX: patched Qwen3-TTS decoder.enable_cudagraph as no-op.")

    if not getattr(CUDAGraphDecoderWrapper, "_metax_warmup_patched", False):

        def _skip_warmup(self, *args, **kwargs):
            logger.warning(
                "MetaX: skipped Qwen3-TTS Code2Wav CUDA Graph warmup; "
                "set VLLM_OMNI_METAX_ENABLE_CODE2WAV_CUDAGRAPH=1 to enable."
            )
            return None

        CUDAGraphDecoderWrapper.warmup = _skip_warmup
        CUDAGraphDecoderWrapper._metax_warmup_patched = True
        logger.warning("MetaX: patched Qwen3-TTS CUDA Graph warmup as no-op.")


def _patch_snakebeta_triton() -> None:
    if os.getenv("VLLM_OMNI_METAX_ENABLE_TRITON_SNAKEBETA", "0") == "1":
        logger.warning("MetaX: Qwen3-TTS Triton SnakeBeta is explicitly enabled.")
        return

    try:
        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            SnakeBeta,
        )
    except Exception:
        logger.debug("MetaX: Qwen3-TTS SnakeBeta patch skipped.", exc_info=True)
        return

    if getattr(SnakeBeta, "_metax_triton_patched", False):
        return

    # Force eager path.
    SnakeBeta._triton_kernel = False
    SnakeBeta._init_triton = staticmethod(lambda: False)
    SnakeBeta._metax_triton_patched = True

    logger.warning(
        "MetaX: disabled Qwen3-TTS Triton SnakeBeta; "
        "set VLLM_OMNI_METAX_ENABLE_TRITON_SNAKEBETA=1 to enable."
    )


def apply_metax_qwen3_tts_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_code2wav_cudagraph()
    _patch_snakebeta_triton()

    _PATCHED = True
