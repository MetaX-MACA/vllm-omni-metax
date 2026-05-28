# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm_omni_metax.patches.rope_patch import apply_rope_patch
from vllm_omni_metax.patches.qwen3_tts_runtime_patch import (
    apply_metax_qwen3_tts_runtime_patches,
)
from vllm_omni_metax.patches.stream_patch import (
    use_current_stream_for_runner_init,
)

__all__ = [
    "apply_rope_patch",
    "apply_metax_qwen3_tts_runtime_patches",
    "use_current_stream_for_runner_init",
]
