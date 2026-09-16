# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""JoyAI-Image-Edit out-of-tree model integration.

Ships the JoyAI-Image-Edit diffusion pipeline (upstream PR #4112) alongside
the MetaX omni plugin, because the installed core ``vllm_omni`` (v0.26.0) does
not yet contain it.  At plugin-activation time the vendored ``joy_image``
source files are written into the installed
``vllm_omni/diffusion/models/joy_image/`` package directory, and the pipeline
is registered in the diffusion model registry / pre-process / post-process /
metadata tables so that ``vllm serve ... --omni`` can resolve it.

On top of the upstream backport, this patch adapts the raw JoyAI checkpoint
layout shipped in the MetaX model registry
(``transformer/transformer.pth`` + ``vae/Wan2.1_VAE.pth`` +
``JoyAI-Image-Und/``) into a Diffusers-format mirror that the pipeline can
load (``joy_image_model_builder``), and applies a runtime fix for the
Qwen3-VL text encoder (``JoyAI-Image-Und``) on transformers 5.14 / MetaX:
the conditioning forward must run with ``use_cache=False``, otherwise the KV
cache update crashes with ``IndexError`` inside ``transformers.cache_utils``.

Registration is lazy + retried: the omni platform plugin activates while
vllm-omni is still initialising, so importing the registry there would hit a
circular import.  We write the source files eagerly (pure filesystem copy),
then poll until the registry module is fully loaded before mutating its
tables.
"""
from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import threading

logger = logging.getLogger(__name__)

_REGISTERED = False
_QWEN3_VL_REGISTERED = False
_FILES_INSTALLED = False
_MAX_RETRIES = 600
_RETRY_COUNT = 0

_VENDORED_FILES = (
    "__init__.py",
    "cfg_parallel.py",
    "joy_image_edit_transformer.py",
    "pipeline_joy_image_edit.py",
)

_MODEL_ARCH = "JoyImageEditPipeline"
_MODULE_NAME = "joy_image"
_CLASS_NAME = "JoyImageEditPipeline"
_PIPELINE_MODULE = "pipeline_joy_image_edit"


def _vendor_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "joy_image")


def _target_dir() -> str:
    import vllm_omni

    return os.path.join(
        os.path.dirname(os.path.abspath(vllm_omni.__file__)),
        "diffusion",
        "models",
        "joy_image",
    )


def _install_joy_image_files() -> None:
    global _FILES_INSTALLED
    if _FILES_INSTALLED:
        return
    vendor = _vendor_dir()
    target = _target_dir()
    os.makedirs(target, exist_ok=True)
    for fname in _VENDORED_FILES:
        src = os.path.join(vendor, fname)
        dst = os.path.join(target, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"vendored joy_image source missing: {src}")
        shutil.copyfile(src, dst)
    _FILES_INSTALLED = True
    logger.info("joy_image: vendored source installed to %s", target)


def _register_pipeline() -> None:
    from vllm_omni.diffusion import registry as reg
    from vllm_omni.diffusion import model_metadata as mm

    if getattr(reg, "_DIFFUSION_MODELS", None) is None:
        raise RuntimeError("registry._DIFFUSION_MODELS not ready yet")

    # _DIFFUSION_MODELS drives pre/post-process module path resolution.
    if _MODEL_ARCH not in reg._DIFFUSION_MODELS:
        reg._DIFFUSION_MODELS[_MODEL_ARCH] = (
            _MODULE_NAME,
            _PIPELINE_MODULE,
            _CLASS_NAME,
        )
    # DiffusionModelRegistry is built once at import time from
    # _DIFFUSION_MODELS, so an out-of-tree entry must be added to its
    # internal `models` dict too, otherwise _try_load_model_cls misses it.
    if _MODEL_ARCH not in reg.DiffusionModelRegistry.models:
        reg.DiffusionModelRegistry.register_model(
            _MODEL_ARCH,
            f"vllm_omni.diffusion.models.{_MODULE_NAME}.{_PIPELINE_MODULE}:{_CLASS_NAME}",
        )
    reg._DIFFUSION_POST_PROCESS_FUNCS.setdefault(
        _MODEL_ARCH, "get_joy_image_edit_post_process_func"
    )
    reg._DIFFUSION_PRE_PROCESS_FUNCS.setdefault(
        _MODEL_ARCH, "get_joy_image_edit_pre_process_func"
    )

    if not hasattr(mm, "JOY_IMAGE_EDIT_MAX_INPUT_IMAGES"):
        mm.JOY_IMAGE_EDIT_MAX_INPUT_IMAGES = 1
    if _MODEL_ARCH not in mm._DIFFUSION_MODEL_METADATA:
        mm._DIFFUSION_MODEL_METADATA[_MODEL_ARCH] = mm.DiffusionModelMetadata(
            supports_multimodal_inputs=True,
            max_multimodal_image_inputs=mm.JOY_IMAGE_EDIT_MAX_INPUT_IMAGES,
        )


def _patch_text_encoder_cache() -> None:
    """Force ``use_cache=False`` for the JoyAI Qwen3-VL conditioning pass.

    ``JoyAI-Image-Und`` ships with ``use_cache=true`` in its config, and the
    upstream pipeline's ``_get_last_layer_pre_norm_hidden`` runs a full
    ``Qwen3VLForConditionalGeneration`` forward to capture the last-layer
    pre-norm hidden states.  On transformers 5.14 (installed in the MetaX
    image) the generated KV cache crashes with
    ``IndexError: Dimension out of range ... in DynamicCache.update``.
    The conditioning pass only needs hidden states, never the cache, so we
    force it off.
    """
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )

    original_get_hidden = JoyImageEditPipeline._get_last_layer_pre_norm_hidden

    def get_hidden_no_cache(self, model_inputs):
        input_items = getattr(model_inputs, "items", None)
        if callable(input_items):
            text_encoder_inputs = dict(input_items())
        else:
            text_encoder_inputs = dict(vars(model_inputs))
        text_encoder_inputs["use_cache"] = False
        return original_get_hidden(self, text_encoder_inputs)

    JoyImageEditPipeline._get_last_layer_pre_norm_hidden = get_hidden_no_cache
    logger.info("joy_image: Qwen3-VL conditioning pass forced to use_cache=False.")


def _patch_decode_gpu_sync() -> None:
    """Force a device sync after JoyAI forward (VAE decode) on MetaX.

    Same root cause as ``wan_sync_patch``: the omni stage can read the VAE
    decode result before the GPU work is finished on the MetaX stack, which
    corrupts the output (banding / partial channels).  Syncing right after
    ``JoyImageEditPipeline.forward`` makes the output deterministic.
    """
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )

    original_forward = JoyImageEditPipeline.forward

    def forward_with_sync(self, req, **kwargs):
        from vllm_omni.platforms import current_omni_platform  # lazy import

        output = original_forward(self, req, **kwargs)
        current_omni_platform.synchronize()
        return output

    JoyImageEditPipeline.forward = forward_with_sync
    logger.info("joy_image: JoyImageEditPipeline.forward wrapped with post-decode GPU sync.")


def _register_qwen3_vl_pipeline() -> None:
    """Register a single-LLM-stage omni pipeline for Qwen3-VL checkpoints.

    ``JoyAI-Image-Und`` (and any Qwen3-VL checkpoint) is a plain VLM.  Without
    a registry entry, vllm-omni's ``--omni`` mode cannot resolve a pipeline and
    falls back to a default *diffusion* stage, which fails with
    ``Model class Qwen3VLForConditionalGeneration not found in diffusion model
    registry``.  Registering ``qwen3_vl`` routes it to a single LLM stage so
    ``vllm serve <JoyAI-Image-Und> --omni`` works like a native vLLM VLM
    service (text + image chat).
    """
    from vllm_omni.config.pipeline_registry import (
        OMNI_PIPELINES,
        register_pipeline,
    )
    from vllm_omni.config.stage_config import (
        PipelineConfig,
        StageExecutionType,
        StagePipelineConfig,
    )

    if "qwen3_vl" in OMNI_PIPELINES:
        logger.info("joy_image: qwen3_vl already registered in OMNI_PIPELINES.")
        return

    pipeline = PipelineConfig(
        model_type="qwen3_vl",
        model_arch="Qwen3VLForConditionalGeneration",
        hf_architectures=("Qwen3VLForConditionalGeneration",),
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="llm",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(),
                final_output=True,
                final_output_type="text",
                owns_tokenizer=True,
                requires_multimodal_data=True,
                model_arch="Qwen3VLForConditionalGeneration",
                engine_output_type="text",
                sampling_constraints={"detokenize": True},
            ),
        ),
    )
    register_pipeline(pipeline, model_type="qwen3_vl")
    logger.info("joy_image: qwen3_vl omni pipeline registered (single LLM stage).")


def _patch_raw_layout_support() -> None:
    """Resolve the raw MetaX JoyAI checkpoint to the Diffusers mirror."""
    from vllm_omni.diffusion import data as diffusion_data
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )
    from vllm_omni_metax.patches.joy_image_model_builder import (
        is_raw_joyai_layout,
        resolve_model_path,
    )

    # 1. Server-side class resolution: raw layout -> JoyImageEditPipeline.
    original_resolve = diffusion_data.resolve_model_class_name

    def resolve_with_joyai(model, diffusion_load_format="default"):
        if is_raw_joyai_layout(model):
            return _MODEL_ARCH
        return original_resolve(model, diffusion_load_format)

    diffusion_data.resolve_model_class_name = resolve_with_joyai

    # 2. Worker-side enrich_config: point self.model at the mirror so
    #    model_index.json / _class_name / metadata resolution all work.
    original_enrich = diffusion_data.OmniDiffusionConfig.enrich_config

    def enrich_with_joyai(self):
        model = getattr(self, "model", None)
        if model:
            self.model = resolve_model_path(model)
        return original_enrich(self)

    diffusion_data.OmniDiffusionConfig.enrich_config = enrich_with_joyai

    # 3. Direct-construction fallback inside the pipeline itself.
    original_init = JoyImageEditPipeline.__init__

    def init_with_joyai(self, *, od_config, prefix=""):
        model = getattr(od_config, "model", None)
        if model:
            od_config.model = resolve_model_path(model)
        return original_init(self, od_config=od_config, prefix=prefix)

    JoyImageEditPipeline.__init__ = init_with_joyai

    # 4. Engine-side stage-config resolution reads the model root config.json;
    #    for the raw JoyAI checkpoint that file has no model_type/model_index,
    #    so resolve the model to the Diffusers mirror before resolving configs.
    try:
        from vllm_omni.entrypoints import utils as entrypoint_utils

        original_resolve_config_path = entrypoint_utils.resolve_model_config_path

        def resolve_config_path_with_joyai(model):
            resolved = resolve_model_path(model)
            return original_resolve_config_path(resolved)

        entrypoint_utils.resolve_model_config_path = resolve_config_path_with_joyai
    except Exception:
        logger.warning(
            "joy_image: could not patch resolve_model_config_path; "
            "serving the raw checkpoint may fail.",
            exc_info=True,
        )
    logger.info("joy_image: raw checkpoint -> Diffusers mirror resolution hooks installed.")


def _register_with_retry() -> None:
    global _REGISTERED, _RETRY_COUNT
    if _REGISTERED:
        return

    # Part 1: qwen3_vl omni pipeline (pure LLM stage). Needed even when no
    # diffusion module is ever imported (e.g. serving JoyAI-Image-Und alone).
    if not _QWEN3_VL_REGISTERED:
        if not _pipeline_registry_settled():
            _retry_register()
            return
        try:
            _register_qwen3_vl_pipeline()
            globals()["_QWEN3_VL_REGISTERED"] = True
        except Exception:
            logger.warning(
                "joy_image: qwen3_vl pipeline registration failed; will retry.",
                exc_info=True,
            )
            _retry_register()
            return

    # Part 2: JoyAI diffusion pipeline registration + MetaX runtime fixes.
    if not _diffusion_registry_settled():
        _retry_register()
        return

    try:
        _register_pipeline()
        _patch_text_encoder_cache()
        _patch_decode_gpu_sync()
        _patch_raw_layout_support()
    except Exception:
        logger.warning(
            "joy_image patch: registration attempt failed; will retry.",
            exc_info=True,
        )
        _retry_register()
        return
    _REGISTERED = True
    logger.info("JoyAI-Image-Edit pipeline registered via vllm-omni-metax.")


def _retry_register() -> None:
    global _RETRY_COUNT
    if _RETRY_COUNT < _MAX_RETRIES:
        _RETRY_COUNT += 1
        threading.Timer(1.0, _register_with_retry).start()
    else:
        logger.error(
            "joy_image patch: vllm_omni never became ready after %d retries.",
            _MAX_RETRIES,
        )


def _pipeline_registry_settled() -> bool:
    """True once vllm_omni's pipeline registry import has settled.

    The omni platform plugin activates while ``vllm_omni`` is still being
    imported, and our retry runs on a timer thread.  Importing
    ``vllm_omni.config.pipeline_registry`` (or other early modules) from that
    thread while the main import is still inside it can return a partially
    initialized module, which then breaks the main thread's own import
    (circular-import failure).  So we only proceed once the module is present
    in ``sys.modules`` *and* fully initialized, and never trigger the import
    ourselves.
    """
    pipeline_registry = sys.modules.get("vllm_omni.config.pipeline_registry")
    if pipeline_registry is None or not hasattr(pipeline_registry, "OMNI_PIPELINES"):
        return False
    return True


def _diffusion_registry_settled() -> bool:
    """True once vllm_omni's diffusion registry/metadata imports settled."""
    diffusion_registry = sys.modules.get("vllm_omni.diffusion.registry")
    if diffusion_registry is None or not hasattr(diffusion_registry, "_DIFFUSION_MODELS"):
        return False
    metadata = sys.modules.get("vllm_omni.diffusion.model_metadata")
    if metadata is None or not hasattr(metadata, "_DIFFUSION_MODEL_METADATA"):
        return False
    return True


def apply_joy_image_patch() -> None:
    if _REGISTERED:
        return
    try:
        _install_joy_image_files()
    except Exception:
        logger.warning("Failed to install joy_image files.", exc_info=True)
        return
    _register_with_retry()
