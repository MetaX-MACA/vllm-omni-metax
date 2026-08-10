"""vllm-omni-metax adapter plugin.

This package intentionally does **not** register any `vllm.platform_plugins`
entry point. The MetaX platform continues to be owned by `vllm-metax`.
It only registers a `vllm_omni.platform_plugins` entry point so that
`vllm-omni` can resolve a MetaX-aware Omni platform.
"""

from .plugin import metax_omni_platform_plugin
from .platform import MetaxOmniPlatform

__all__ = ["metax_omni_platform_plugin", "MetaxOmniPlatform"]

