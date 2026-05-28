from __future__ import annotations

from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni_metax.patches.stream_patch import use_current_stream_for_runner_init


class MetaxGPUARWorker(GPUARWorker):
    def init_device(self):
        with use_current_stream_for_runner_init():
            return super().init_device()
