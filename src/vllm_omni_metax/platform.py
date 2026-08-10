# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import os

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm_metax.platform import MacaPlatform

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class MetaxOmniPlatform(OmniPlatform, MacaPlatform):
    _omni_enum = OmniPlatformEnum.CUDA
    _enum = OmniPlatformEnum.CUDA

    device_enum = OmniPlatformEnum.CUDA
    device_type = "cuda"
    dist_backend = "nccl"
    device_control_env_var = "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Deploy-patch lazy install.  Called from worker-cls getters which
    # are invoked during stage engine init — well after all imports are
    # complete and stage_config is guaranteed fully loaded.
    # ------------------------------------------------------------------

    _deploy_patch_done: bool = False

    @classmethod
    def _ensure_deploy_patch(cls) -> None:
        if cls._deploy_patch_done:
            return
        cls._deploy_patch_done = True
        try:
            from vllm_omni_metax.patches.deploy_resolution_patch import ensure_patch_installed

            ensure_patch_installed()
        except Exception:
            pass

    # ------------------------------------------------------------------

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        cls._ensure_deploy_patch()
        return "vllm_omni_metax.worker.gpu_ar_worker.MetaxGPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        cls._ensure_deploy_patch()
        return "vllm_omni_metax.worker.gpu_generation_worker.MetaxGPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        # Aligned with the rel0260 CudaOmniPlatform: deploy YAMLs live under
        # vllm_omni/deploy/.  The plugin deploy/ directory takes precedence
        # via deploy_resolution_patch (see patches/).
        return "vllm_omni/deploy"

    @classmethod
    def has_flash_attn_package(cls) -> bool:
        return False

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
        )
        from vllm_omni.diffusion.envs import PACKAGES_CHECKER

        # MetaX only supports FLASH_ATTN (when hardware + packages allow it)
        # and TORCH_SDPA.  All other backends (FLASH_ATTN_HUB, FLASH_ATTN_3_HUB,
        # SAGE_ATTN(_3), CUDNN_ATTN, FLASHINFER_ATTN, TRTLLM_ATTN) require
        # NVIDIA-specific kernels and are not available on MACA.
        _unsupported_backends = {
            "FLASH_ATTN_HUB",
            "FLASH_ATTN_3_HUB",
            "SAGE_ATTN",
            "SAGE_ATTN_3",
            "CUDNN_ATTN",
            "FLASHINFER_ATTN",
            "TRTLLM_ATTN",
        }

        # Check compute capability for Flash Attention support
        # Flash Attention requires compute capability >= 8.0 and < 10.0
        compute_capability = cls.get_device_capability()
        compute_supported = False
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = 80 <= capability < 100

        # Check if FA packages are available
        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)

        # Both compute capability and packages must be available for FA
        flash_attn_supported = compute_supported and packages_available

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper in _unsupported_backends:
                logger.warning(
                    "Diffusion attention backend '%s' is not available on MetaX "
                    "hardware. Falling back to TORCH_SDPA backend.",
                    backend_upper,
                )
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention requires GPU with compute capability >= 8.0 "
                        "and < 10.0. Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning("Flash Attention packages not available. Falling back to TORCH_SDPA backend.")
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def set_device(cls, device: torch.device | int) -> None:
        if isinstance(device, torch.device):
            if device.index is not None:
                torch.cuda.set_device(device.index)
            else:
                torch.cuda.set_device(device)
        else:
            torch.cuda.set_device(device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        except Exception as e:
            logger.warning(
                "Failed to get device capability: %s",
                e,
            )
            return None

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_version(cls) -> str | None:
        return getattr(torch.version, "cuda", None)

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return True

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        value = "" if devices is None else str(devices)

        os.environ["CUDA_VISIBLE_DEVICES"] = value
        os.environ["MACA_VISIBLE_DEVICES"] = value

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("MACA_VISIBLE_DEVICES", None)

    @classmethod
    def is_cuda(cls) -> bool:
        # Treat MetaX as omni's GPU/CUDA-like backend.
        return True

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_omni.profiler.omni_torch_profiler.OmniTorchProfilerWrapper"
