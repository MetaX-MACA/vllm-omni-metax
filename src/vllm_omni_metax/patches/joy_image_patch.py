# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""JoyAI-Image-Edit out-of-tree model integration.

Ships the JoyAI-Image-Edit diffusion pipeline (upstream PR #4112) alongside
the MetaX omni plugin, because the installed core ``vllm_omni`` (v0.30.0) does
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
_REGISTER_LOCK = threading.RLock()
_RETRY_TIMER = None

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
    """Force use_cache=False for the JoyAI Qwen3-VL conditioning pass."""
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )

    original_get_hidden = JoyImageEditPipeline._get_last_layer_pre_norm_hidden
    if getattr(original_get_hidden, "_metax_joyai_no_cache", False):
        return

    def get_hidden_no_cache(self, model_inputs):
        input_items = getattr(model_inputs, "items", None)
        if callable(input_items):
            text_encoder_inputs = dict(input_items())
        else:
            text_encoder_inputs = dict(vars(model_inputs))
        text_encoder_inputs["use_cache"] = False
        return original_get_hidden(self, text_encoder_inputs)

    get_hidden_no_cache._metax_joyai_no_cache = True
    JoyImageEditPipeline._get_last_layer_pre_norm_hidden = get_hidden_no_cache
    logger.info("joy_image: Qwen3-VL conditioning pass forced to use_cache=False.")


def _patch_decode_gpu_sync() -> None:
    """Synchronize after the JoyAI pipeline forward pass on MetaX."""
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )

    original_forward = JoyImageEditPipeline.forward
    if getattr(original_forward, "_metax_joyai_sync", False):
        return

    def forward_with_sync(self, req, **kwargs):
        from vllm_omni.platforms import current_omni_platform

        output = original_forward(self, req, **kwargs)
        current_omni_platform.synchronize()
        return output

    forward_with_sync._metax_joyai_sync = True
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

    original_resolve = diffusion_data.resolve_model_class_name
    if getattr(original_resolve, "_metax_joyai_resolve", False):
        resolve_with_joyai = original_resolve
        original_resolve = resolve_with_joyai._metax_joyai_original_resolve
    else:
        def resolve_with_joyai(model, diffusion_load_format="default", revision=None):
            if is_raw_joyai_layout(model):
                return _MODEL_ARCH
            return original_resolve(model, diffusion_load_format, revision)

        resolve_with_joyai._metax_joyai_resolve = True
        resolve_with_joyai._metax_joyai_original_resolve = original_resolve
        diffusion_data.resolve_model_class_name = resolve_with_joyai

    resolver = sys.modules.get("vllm_omni.config.resolver")
    if (
        resolver is not None
        and getattr(resolver, "resolve_model_class_name", None) is original_resolve
    ):
        resolver.resolve_model_class_name = resolve_with_joyai

    original_enrich = diffusion_data.OmniDiffusionConfig.enrich_config
    if not getattr(original_enrich, "_metax_joyai_enrich", False):
        def enrich_with_joyai(self):
            model = getattr(self, "model", None)
            if model:
                self.model = resolve_model_path(model)
            return original_enrich(self)

        enrich_with_joyai._metax_joyai_enrich = True
        diffusion_data.OmniDiffusionConfig.enrich_config = enrich_with_joyai

    original_init = JoyImageEditPipeline.__init__
    if not getattr(original_init, "_metax_joyai_init", False):
        def init_with_joyai(self, *, od_config, prefix=""):
            model = getattr(od_config, "model", None)
            if model:
                od_config.model = resolve_model_path(model)
            return original_init(self, od_config=od_config, prefix=prefix)

        init_with_joyai._metax_joyai_init = True
        JoyImageEditPipeline.__init__ = init_with_joyai


def _register_with_retry() -> None:
    global _REGISTERED, _QWEN3_VL_REGISTERED
    with _REGISTER_LOCK:
        if _REGISTERED:
            return

        if not _QWEN3_VL_REGISTERED:
            pipeline_registry = sys.modules.get("vllm_omni.config.pipeline_registry")
            if not _pipeline_registry_settled() or not callable(
                getattr(pipeline_registry, "register_pipeline", None)
            ):
                _retry_register()
                return
            try:
                _register_qwen3_vl_pipeline()
                _QWEN3_VL_REGISTERED = True
            except Exception:
                logger.warning(
                    "joy_image: qwen3_vl pipeline registration failed; will retry.",
                    exc_info=True,
                )
                _retry_register()
                return

        diffusion_registry = sys.modules.get("vllm_omni.diffusion.registry")
        if not _diffusion_registry_settled() or not hasattr(
            diffusion_registry, "DiffusionModelRegistry"
        ):
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
    global _RETRY_COUNT, _RETRY_TIMER
    with _REGISTER_LOCK:
        if _REGISTERED or _RETRY_TIMER is not None:
            return
        if _RETRY_COUNT >= _MAX_RETRIES:
            logger.error(
                "joy_image patch: vllm_omni never became ready after %d retries.",
                _MAX_RETRIES,
            )
            return
        _RETRY_COUNT += 1
        timer = threading.Timer(1.0, _retry_register_once)
        timer.daemon = True
        _RETRY_TIMER = timer
        timer.start()


def _retry_register_once() -> None:
    global _RETRY_TIMER
    with _REGISTER_LOCK:
        _RETRY_TIMER = None
    _register_with_retry()


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

