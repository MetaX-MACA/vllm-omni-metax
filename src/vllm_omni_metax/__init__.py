# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""vllm-omni-metax adapter plugin.

Keep package import lightweight. Entry points import ``vllm_omni_metax.plugin``
directly, and model inspection may import ``vllm_omni_metax.models`` in a
subprocess. Importing platform classes here can trigger circular vllm-omni
platform resolution during model inspection.
"""

from .plugin import metax_omni_platform_plugin

__all__ = ["metax_omni_platform_plugin"]

