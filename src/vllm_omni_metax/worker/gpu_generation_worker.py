from __future__ import annotations

from vllm.logger import init_logger
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker
from vllm_omni_metax.patches.stream_patch import use_current_stream_for_runner_init

logger = init_logger(__name__)


class MetaxGPUGenerationWorker(GPUGenerationWorker):
    def _apply_metax_runtime_patches(self) -> None:
        from vllm_omni_metax.patches import apply_metax_qwen3_tts_runtime_patches

        logger.info("MetaX: applying generation worker runtime patches.")
        apply_metax_qwen3_tts_runtime_patches()

    def init_device(self):
        self._apply_metax_runtime_patches()
        with use_current_stream_for_runner_init():
            return super().init_device()

    def load_model(self, *args, **kwargs):
        self._apply_metax_runtime_patches()
        return super().load_model(*args, **kwargs)
