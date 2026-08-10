# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Monkey-patch deploy YAML resolution so the plugin can ship its own deploy configs.

vllm-omni 0.26 resolves deploy YAMLs through
``vllm_omni.config.config_factory.StageConfigFactory``:
- ``create_from_model`` — structured ``VllmOmniConfig`` path
- ``create_legacy_stage_configs_from_model`` — current runtime legacy path

Both receive ``deploy_config_path``, which is ``None`` when the user did not
pass ``--deploy-config`` (see ``vllm_omni/entrypoints/utils.py``).
This patch inserts the plugin's deploy directory ahead of the builtin
``vllm_omni/deploy/`` by resolving a plugin YAML for the model when
``deploy_config_path is None``.

**Deferred installation.**  During platform plugin loading the
``vllm_omni.config.config_factory`` module may still be initialising, which
makes a direct ``from ... import StageConfigFactory`` trigger a circular
import.  We therefore look up ``StageConfigFactory`` via ``sys.modules`` and
install the real monkey-patch only once the class is fully available.  If
the class is not yet available when ``apply_deploy_resolution_patch()``
runs, the patch is retried on the first worker-cls getter call (via
``MetaxOmniPlatform._ensure_deploy_patch``), i.e. at stage-engine init time,
when the module is guaranteed to be fully loaded.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ATTEMPTED = False
_INSTALLED = False
_WARNING_LOGGED = False

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"

_orig_create_from_model = None
_orig_create_legacy = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stage_config_factory():
    """Return StageConfigFactory without triggering a fresh import.

    Uses ``sys.modules`` to avoid the circular import that occurs when
    ``from vllm_omni.config.config_factory import StageConfigFactory`` is
    executed while the module is still being initialised.
    """
    mod = sys.modules.get("vllm_omni.config.config_factory")
    if mod is not None:
        return getattr(mod, "StageConfigFactory", None)
    # Module not yet imported at all — safe to import normally.
    try:
        from vllm_omni.config.config_factory import StageConfigFactory

        return StageConfigFactory
    except ImportError:
        return None


def _resolve_plugin_deploy_path(
    factory_cls,
    model: str,
    trust_remote_code: bool | None,
    deploy_config_path: str | None,
) -> str | None:
    """Return the plugin deploy YAML path when it exists for the model.

    Precedence (highest to lowest):
    1. ``--deploy-config`` CLI flag (untouched, ``deploy_config_path`` set)
    2. Plugin ``deploy/`` directory (this function)
    3. Builtin ``vllm_omni/deploy/`` / registry default (upstream resolution)

    ``try_infer_model_type`` is cached upstream, so inferring the HF model
    type here costs nothing extra (``get_pipeline_config`` uses the same
    cached result).
    """
    if deploy_config_path is not None:
        return deploy_config_path
    try:
        # HF config resolution needs a real bool: transformers treats None
        # as "prompt for consent", which blocks non-interactively.
        model_type = factory_cls.try_infer_model_type(model, bool(trust_remote_code))
    except Exception:
        logger.debug(
            "MetaX: model type inference failed; skipping plugin deploy "
            "resolution.",
            exc_info=True,
        )
        return deploy_config_path
    if model_type:
        candidate = _DEPLOY_DIR / f"{model_type}.yaml"
        if candidate.exists():
            logger.info("Using plugin deploy YAML: %s", candidate)
            return str(candidate)
        logger.debug(
            "No plugin deploy YAML for %r at %s; falling back to builtin.",
            model_type,
            candidate,
        )
    return deploy_config_path


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _install_factory_patch() -> bool:
    """Install the monkey-patch on StageConfigFactory entry points.

    Returns ``True`` on success, ``False`` when the class is not yet
    available (module still initialising).
    """
    global _orig_create_from_model, _orig_create_legacy

    if _orig_create_from_model is not None:
        return True  # already installed

    StageConfigFactory = _get_stage_config_factory()
    if StageConfigFactory is None:
        return False

    # The originals are classmethods captured bound to the class; the
    # descriptor auto-binds cls, so the wrappers must not pass cls through.
    _orig_create_from_model = StageConfigFactory.create_from_model
    _orig_create_legacy = StageConfigFactory.create_legacy_stage_configs_from_model

    @classmethod
    def _patched_create_from_model(cls, model, *, trust_remote_code,
                                   cli_overrides, deploy_config_path):
        deploy_config_path = _resolve_plugin_deploy_path(
            cls, model, trust_remote_code, deploy_config_path)
        return _orig_create_from_model(
            model,
            trust_remote_code=trust_remote_code,
            cli_overrides=cli_overrides,
            deploy_config_path=deploy_config_path,
        )

    @classmethod
    def _patched_create_legacy(cls, model, *, trust_remote_code,
                               cli_overrides, deploy_config_path,
                               strategy_specs=None):
        deploy_config_path = _resolve_plugin_deploy_path(
            cls, model, trust_remote_code, deploy_config_path)
        return _orig_create_legacy(
            model,
            trust_remote_code=trust_remote_code,
            cli_overrides=cli_overrides,
            deploy_config_path=deploy_config_path,
            strategy_specs=strategy_specs,
        )

    StageConfigFactory.create_from_model = _patched_create_from_model
    StageConfigFactory.create_legacy_stage_configs_from_model = _patched_create_legacy
    StageConfigFactory._metax_deploy_patched = True
    return True


def _try_install_now() -> bool:
    """Attempt installation; return True if successful."""
    global _INSTALLED

    if _INSTALLED:
        return True

    if _install_factory_patch():
        _INSTALLED = True
        logger.info("Deploy resolution patched successfully.")
        return True

    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_deploy_resolution_patch() -> None:
    """Insert the plugin's deploy directory into vllm-omni's YAML resolution.

    Called automatically when the MetaX platform plugin activates.

    If ``StageConfigFactory`` is not yet available (module still
    initialising during platform-plugin loading), the patch is deferred
    and will be retried transparently via ``ensure_patch_installed()``.
    """
    global _ATTEMPTED

    if _ATTEMPTED:
        return
    _ATTEMPTED = True

    # List available YAMLs for diagnostics
    yamls: list[str] = []
    if _DEPLOY_DIR.is_dir():
        yamls = sorted(p.name for p in _DEPLOY_DIR.glob("*.yaml"))
    logger.info(
        "Deploy resolution: plugin dir=%s, available YAMLs=%s",
        _DEPLOY_DIR, yamls if yamls else "<none>",
    )

    if not yamls:
        logger.warning("No deploy YAMLs found in plugin dir; patch skipped.")
        return

    _try_install_now()


# ---------------------------------------------------------------------------
# Lazy retry hook — called from MetaxOmniPlatform._ensure_deploy_patch
# (worker-cls getter time, when config_factory is fully initialised).
# ---------------------------------------------------------------------------


def ensure_patch_installed() -> None:
    """Re-attempt install if it was deferred earlier.

    Safe to call at any time; no-op when already installed or not needed.
    """
    global _ATTEMPTED, _INSTALLED, _WARNING_LOGGED

    if not _ATTEMPTED:
        return  # apply_deploy_resolution_patch was never called
    if _INSTALLED:
        return

    if _try_install_now():
        logger.info("Deploy resolution patch installed (deferred).")
    elif not _WARNING_LOGGED:
        _WARNING_LOGGED = True
        logger.warning(
            "Deploy resolution patch could not be installed. "
            "Use --deploy-config to specify a deploy YAML manually."
        )
