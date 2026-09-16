# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Clean partial-import stubs before vllm optional-dependency probing.

``vllm.utils.import_utils._has_module()`` calls ``importlib.util.find_spec()``,
which raises ``ValueError: <module>.__spec__ is None`` when an optional module
(e.g. ``humming`` on stacks where it is not installed) is left in
``sys.modules`` with ``__spec__ = None``.  vllm then logs
``Module humming was found but failed to import`` at every startup.

Dropping such stubs before the probe makes ``find_spec`` return ``None`` and
keeps the startup log clean.
"""

from __future__ import annotations

import logging
import sys
from vllm.utils import import_utils

logger = logging.getLogger(__name__)

_PATCHED = False


def _patched_has_module(module_name: str) -> bool:
    stub = sys.modules.get(module_name)
    if stub is not None and getattr(stub, "__spec__", None) is None:
        sys.modules.pop(module_name, None)
    return _ORIGINAL_HAS_MODULE(module_name)


_ORIGINAL_HAS_MODULE = import_utils._has_module


def apply_import_utils_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import_utils._has_module = _patched_has_module
    _PATCHED = True
    logger.info("Patched vllm import_utils._has_module to clean partial-import stubs.")
