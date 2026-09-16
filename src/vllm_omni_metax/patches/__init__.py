# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm_omni_metax.patches.code_predictor_patch import apply_code_predictor_patch
from vllm_omni_metax.patches.cudnn_patch import apply_cudnn_patch
from vllm_omni_metax.patches.deploy_resolution_patch import apply_deploy_resolution_patch
from vllm_omni_metax.patches.import_utils_patch import apply_import_utils_patch
from vllm_omni_metax.patches.joy_image_patch import apply_joy_image_patch
from vllm_omni_metax.patches.rope_patch import apply_rope_patch
from vllm_omni_metax.patches.sdpa_min_mask_patch import apply_sdpa_min_mask_patch
from vllm_omni_metax.patches.qwen3_omni_thinker_only_patch import (
    apply_qwen3_omni_thinker_only_patch,
)
from vllm_omni_metax.patches.qwen3_tts_runtime_patch import (
    apply_metax_qwen3_tts_runtime_patches,
)
from vllm_omni_metax.patches.stream_patch import (
    use_current_stream_for_runner_init,
)
from vllm_omni_metax.patches.wan_sync_patch import apply_wan_sync_patch

__all__ = [
    "apply_code_predictor_patch",
    "apply_cudnn_patch",
    "apply_deploy_resolution_patch",
    "apply_import_utils_patch",
    "apply_joy_image_patch",
    "apply_rope_patch",
    "apply_sdpa_min_mask_patch",
    "apply_qwen3_omni_thinker_only_patch",
    "apply_metax_qwen3_tts_runtime_patches",
    "apply_wan_sync_patch",
    "use_current_stream_for_runner_init",
]
