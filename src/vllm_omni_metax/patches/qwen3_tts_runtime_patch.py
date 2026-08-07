# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import threading
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


def _patch_qwen3_tts_base_task_guard(_retries: int = 5) -> None:
    """Qwen3-TTS: fail fast when a 'Base' voice-clone request hits a model
    variant that ships no trained speaker encoder (CustomVoice / VoiceDesign).

    Upstream vllm-omni default-constructs a random speaker encoder
    (Qwen3TTSSpeakerEncoderConfig.enc_dim defaults to 1024) whenever
    config.speaker_encoder_config is absent, then crashes deep inside
    prompt_embeds_builder.build_prompt_embeds() with an obscure
    ``torch.cat`` RuntimeError (speaker_embed width != codec/talker width,
    e.g. 1024 vs 2048). Patch ``preprocess`` to raise an actionable error
    instead.

    Added: V0.22.0 / V0.26.0.
    remove_at: upstream vllm-omni validates task_type against the model
    variant (or projects speaker embeddings into the talker hidden size).
    """
    if os.getenv("VLLM_OMNI_METAX_DISABLE_TTS_BASE_TASK_GUARD", "0") == "1":
        logger.warning("MetaX: Qwen3-TTS Base-task guard is explicitly disabled.")
        return

    try:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )
    except Exception:
        logger.debug("MetaX: Qwen3-TTS Base-task guard patch skipped.", exc_info=True)
        if _retries > 0:
            # The talker module is not importable yet while vllm-omni is still
            # initializing (plugin activation runs mid-import); retry once
            # startup settles, mirroring the deploy-resolution patch retry.
            threading.Timer(2.0, lambda: _patch_qwen3_tts_base_task_guard(_retries - 1)).start()
        return

    if getattr(Qwen3TTSTalkerForConditionalGeneration, "_metax_base_task_guard_patched", False):
        return

    _orig_preprocess = Qwen3TTSTalkerForConditionalGeneration.preprocess

    def _preprocess_with_base_task_guard(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info_dict: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        task_type_raw = info_dict.get("task_type")
        if isinstance(task_type_raw, (list, tuple)) and task_type_raw:
            task_type = task_type_raw[0]
        else:
            task_type = task_type_raw
        if task_type == "Base":
            config = self.config
            model_type = str(getattr(config, "tts_model_type", "") or "").lower()
            talker_config = getattr(config, "talker_config", None)
            hidden_size = int(getattr(talker_config, "hidden_size", 0) or 0)
            spk_cfg = getattr(config, "speaker_encoder_config", None)
            enc_dim = int(getattr(spk_cfg, "enc_dim", 0) or 0)
            dim_mismatch = enc_dim and hidden_size and enc_dim != hidden_size
            if model_type not in ("", "base") or dim_mismatch:
                raise ValueError(
                    "Qwen3-TTS 'Base' task (voice cloning via ref_audio/speaker_embedding) is not supported "
                    f"by this model variant (tts_model_type={model_type!r}, speaker encoder "
                    f"enc_dim={enc_dim}, talker hidden_size={hidden_size}). The checkpoint contains no "
                    "trained speaker encoder. Use a Qwen3-TTS-12Hz-*-Base model for voice cloning, or use "
                    "task_type='CustomVoice' with one of this model's built-in speakers."
                )
        return _orig_preprocess(self, input_ids, input_embeds, **info_dict)

    Qwen3TTSTalkerForConditionalGeneration.preprocess = _preprocess_with_base_task_guard
    Qwen3TTSTalkerForConditionalGeneration._metax_base_task_guard_patched = True
    logger.warning("MetaX: patched Qwen3-TTS preprocess with Base-task variant guard.")


def apply_metax_qwen3_tts_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_code2wav_cudagraph()
    _patch_snakebeta_triton()
    _patch_qwen3_tts_base_task_guard()

    _PATCHED = True
