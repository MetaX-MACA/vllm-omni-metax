# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Monkey-patch deploy YAML resolution so the plugin can ship its own deploy configs.

vllm-omni resolves deploy YAMLs from ``vllm_omni/deploy/<model_type>.yaml`` by
default.  This patch inserts the plugin's deploy directory ahead of the builtin
one so that MetaX-tuned deploy configs are picked up automatically without
requiring ``--deploy-config`` on every invocation.

**Deferred installation.**  During platform plugin loading the
``vllm_omni.config.stage_config`` module may still be initialising, which
makes a direct ``from ... import StageConfigFactory`` trigger a circular
import.  We therefore look up ``StageConfigFactory`` via ``sys.modules`` and
install the real monkey-patch only once the class is fully available.  If
the class is not yet available when ``apply_deploy_resolution_patch()``
runs, the patch is retried on the first call to ``create_from_model`` (i.e.
at model-load time, when the module is guaranteed to be fully loaded).
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

_orig_create_from_registry = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stage_config_factory():
    """Return StageConfigFactory without triggering a fresh import.

    Uses ``sys.modules`` to avoid the circular import that occurs when
    ``from vllm_omni.config.stage_config import StageConfigFactory`` is
    executed while the module is still being initialised.
    """
    mod = sys.modules.get("vllm_omni.config.stage_config")
    if mod is not None:
        return getattr(mod, "StageConfigFactory", None)
    # Module not yet imported at all — safe to import normally.
    try:
        from vllm_omni.config.stage_config import StageConfigFactory

        return StageConfigFactory
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _install_factory_patch() -> bool:
    """Install the monkey-patch on StageConfigFactory._create_from_registry.

    Returns ``True`` on success, ``False`` when the class is not yet
    available (module still initialising).
    """
    global _orig_create_from_registry

    if _orig_create_from_registry is not None:
        return True  # already installed

    StageConfigFactory = _get_stage_config_factory()
    if StageConfigFactory is None:
        return False

    _orig_create_from_registry = StageConfigFactory._create_from_registry

    @classmethod
    def _patched_create(cls, model_type, cli_overrides,
                        deploy_config_path=None):
        if deploy_config_path is None:
            candidate = _DEPLOY_DIR / f"{model_type}.yaml"
            if candidate.exists():
                logger.info("Using plugin deploy YAML: %s", candidate)
                deploy_config_path = str(candidate)
            else:
                logger.debug(
                    "No plugin deploy YAML for %r at %s; falling back to builtin.",
                    model_type, candidate,
                )
        # _orig_create_from_registry is a classmethod — the descriptor
        # auto-binds cls, so we must not pass cls explicitly.
        return _orig_create_from_registry(
            model_type, cli_overrides,
            deploy_config_path=deploy_config_path)

    StageConfigFactory._create_from_registry = _patched_create
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

    Resolution precedence (highest to lowest):
    1. ``--deploy-config`` CLI flag (untouched)
    2. Plugin ``deploy/`` directory
    3. Builtin ``vllm_omni/deploy/``

    If ``StageConfigFactory`` is not yet available (module still
    initialising during platform-plugin loading), the patch is deferred
    and will be retried transparently on the first call to
    ``create_from_model``.
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
# Lazy retry hook — called when _create_from_registry is first invoked
# (after model loading, when stage_config is fully initialised).
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
