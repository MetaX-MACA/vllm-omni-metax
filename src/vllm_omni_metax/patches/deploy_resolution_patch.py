# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Prefer plugin deploy YAMLs over upstream defaults."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ATTEMPTED = False
_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
_LOCK = threading.RLock()


class _MetaxDeployDirectory:
    def __init__(self, upstream):
        self.upstream = Path(upstream)

    def __truediv__(self, name):
        name = Path(name)
        if name.is_absolute():
            return name

        candidate = _DEPLOY_DIR / name
        if candidate.is_file():
            logger.info("Using plugin deploy YAML: %s", candidate)
            return candidate

        return self.upstream / name

    def __fspath__(self):
        return str(self.upstream)

    def __str__(self):
        return str(self.upstream)

    def __getattr__(self, name):
        return getattr(self.upstream, name)


def _install_default_deploy_directory() -> None:
    with _LOCK:
        for name in (
            "vllm_omni.config.stage_config",
            "vllm_omni.config.config_factory",
        ):
            module = sys.modules.get(name)
            if module is None:
                continue

            original = vars(module).get("_DEPLOY_DIR")
            if original is None or isinstance(original, _MetaxDeployDirectory):
                continue

            module._DEPLOY_DIR = _MetaxDeployDirectory(original)
            logger.info("MetaX deploy directory installed in %s", name)


def apply_deploy_resolution_patch() -> None:
    global _ATTEMPTED

    if _ATTEMPTED:
        return

    _ATTEMPTED = True

    if not _DEPLOY_DIR.is_dir() or not any(_DEPLOY_DIR.glob("*.yaml")):
        logger.warning("No deploy YAMLs found in plugin dir: %s", _DEPLOY_DIR)
        return

    _install_default_deploy_directory()


def ensure_patch_installed() -> None:
    if _ATTEMPTED:
        _install_default_deploy_directory()
