# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Build a Diffusers-format JoyAI-Image-Edit mirror from the raw checkpoint
layout shipped in the MetaX model registry.

The upstream vLLM-Omni JoyImageEditPipeline (vllm-omni PR #4112) loads the
model from a Diffusers-style directory (``model_index.json`` +
``transformer/`` + ``vae/`` + ``scheduler/`` + ``text_encoder/`` +
``tokenizer/`` + ``processor/``).  The MetaX registry instead stores the
original JoyAI checkpoint:

    <raw_root>/transformer/transformer.pth     # keys already match the
                                               # vLLM-Omni transformer module
    <raw_root>/vae/Wan2.1_VAE.pth              # Wan2.1 raw VAE key naming
    <raw_root>/JoyAI-Image-Und/                # Qwen3VL text encoder + tokenizer

This module converts/links those pieces into a writable Diffusers mirror the
first time the model is served, and is idempotent afterwards.  The mirror
location can be overridden with ``VLLM_OMNI_METAX_JOYAI_MIRROR_DIR``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

_MIRROR_ENV = "VLLM_OMNI_METAX_JOYAI_MIRROR_DIR"
_TRANSFORMER_SHARD_BYTES = int(4.6 * 1024**3)


TRANSFORMER_CONFIG = {
    "_class_name": "JoyImageEditTransformer3DModel",
    "_diffusers_version": "0.38.0.dev0",
    "hidden_size": 4096,
    "in_channels": 16,
    "mlp_width_ratio": 4.0,
    "num_attention_heads": 32,
    "num_layers": 40,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_dim_list": [16, 56, 56],
    "rope_type": "rope",
    "text_dim": 4096,
    "theta": 10000,
}

VAE_CONFIG = {
    "_class_name": "AutoencoderKLWan",
    "_diffusers_version": "0.38.0.dev0",
    "attn_scales": [],
    "base_dim": 96,
    "decoder_base_dim": None,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 3,
    "is_residual": False,
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ],
    "num_res_blocks": 2,
    "out_channels": 3,
    "patch_size": None,
    "scale_factor_spatial": 8,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 16,
}

SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.38.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.5,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

MODEL_INDEX = {
    "_class_name": "JoyImageEditPipeline",
    "_diffusers_version": "0.38.0.dev0",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "transformer": ["diffusers", "JoyImageEditTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def is_raw_joyai_layout(model: str | None) -> bool:
    """True when *model* points at the raw JoyAI checkpoint layout."""
    if not model or not os.path.isdir(model):
        return False
    return (
        os.path.isfile(os.path.join(model, "transformer", "transformer.pth"))
        and os.path.isfile(os.path.join(model, "vae", "Wan2.1_VAE.pth"))
        and os.path.isdir(os.path.join(model, "JoyAI-Image-Und"))
    )


def default_mirror_dir(raw_root: str) -> str:
    env_dir = os.environ.get(_MIRROR_ENV)
    if env_dir:
        return env_dir
    sibling = os.path.join(os.path.dirname(raw_root.rstrip(os.sep)), "JoyAI-Image-Edit-Diffusers")
    try:
        os.makedirs(sibling, exist_ok=True)
        probe = os.path.join(sibling, ".write_check")
        with open(probe, "a"):
            pass
        os.remove(probe)
        return sibling
    except OSError:
        logger.warning(
            "JoyAI mirror dir %s is not writable; falling back to user cache.",
            sibling,
        )
    return os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "vllm-omni-metax",
        "JoyAI-Image-Edit-Diffusers",
    )


def resolve_model_path(model: str | None) -> str | None:
    """Return the Diffusers mirror path when *model* is a raw JoyAI layout,
    otherwise return *model* unchanged."""
    if not is_raw_joyai_layout(model):
        return model
    mirror = default_mirror_dir(model)
    ensure_mirror(model, mirror_dir=mirror)
    return mirror


@contextlib.contextmanager
def _mirror_lock(raw_root: str) -> Iterator[None]:
    lock_path = os.path.join(default_mirror_dir(raw_root), ".mirror.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _map_resnet_tail(rest: list[str]) -> str:
    if rest and rest[0] == "residual":
        rest = rest[1:]
    head = rest[0]
    if head == "0":
        return "norm1." + rest[1]
    if head == "2":
        return "conv1." + rest[1]
    if head == "3":
        return "norm2." + rest[1]
    if head == "6":
        return "conv2." + rest[1]
    if head == "shortcut":
        return "conv_shortcut." + rest[1]
    return ".".join(rest)


def map_vae_key(raw: str) -> str:
    parts = raw.split(".")
    if raw.startswith("encoder.conv1."):
        return "encoder.conv_in." + parts[2]
    if raw.startswith("encoder.downsamples."):
        return f"encoder.down_blocks.{parts[2]}." + _map_resnet_tail(parts[3:])
    if raw.startswith("encoder.middle."):
        idx = parts[2]
        if idx == "0":
            return "encoder.mid_block.resnets.0." + _map_resnet_tail(parts[3:])
        if idx == "1":
            return "encoder.mid_block.attentions.0." + ".".join(parts[3:])
        if idx == "2":
            return "encoder.mid_block.resnets.1." + _map_resnet_tail(parts[3:])
    if raw.startswith("encoder.head."):
        if parts[2] == "0":
            return "encoder.norm_out.gamma"
        if parts[2] == "2":
            return "encoder.conv_out." + parts[3]
    if raw.startswith("conv1."):
        return "quant_conv." + parts[1]
    if raw.startswith("conv2."):
        return "post_quant_conv." + parts[1]
    if raw.startswith("decoder.conv1."):
        return "decoder.conv_in." + parts[2]
    if raw.startswith("decoder.middle."):
        idx = parts[2]
        if idx == "0":
            return "decoder.mid_block.resnets.0." + _map_resnet_tail(parts[3:])
        if idx == "1":
            return "decoder.mid_block.attentions.0." + ".".join(parts[3:])
        if idx == "2":
            return "decoder.mid_block.resnets.1." + _map_resnet_tail(parts[3:])
    if raw.startswith("decoder.upsamples."):
        block = int(parts[2]) // 4
        within = int(parts[2]) % 4
        if within == 3:
            return f"decoder.up_blocks.{block}.upsamplers.0." + ".".join(parts[3:])
        return f"decoder.up_blocks.{block}.resnets.{within}." + _map_resnet_tail(parts[3:])
    if raw.startswith("decoder.head."):
        if parts[2] == "0":
            return "decoder.norm_out.gamma"
        if parts[2] == "2":
            return "decoder.conv_out." + parts[3]
    raise KeyError(f"Unmapped raw VAE key: {raw}")


def _load_raw_state_dict(path: str) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _remap_transformer_key(name: str) -> str:
    """Insert the ``.attn.`` level used by the Diffusers / vLLM-Omni port.

    The raw JoyAI checkpoint stores joint attention weights directly under the
    double block (``double_blocks.N.img_attn_qkv.weight``), while the
    Diffusers ``JoyImageEditTransformer3DModel`` / vLLM-Omni port nests them
    under ``double_blocks.N.attn.*``.  Everything else already matches.
    """
    match = re.match(r"^(double_blocks\.\d+)\.(img_attn_|txt_attn_)", name)
    if match:
        return f"{match.group(1)}.attn.{match.group(2)}" + name[len(match.group(0)):]
    return name


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _mirror_complete(mirror: str) -> bool:
    return (
        os.path.isfile(os.path.join(mirror, "model_index.json"))
        and os.path.isfile(os.path.join(mirror, "transformer", "config.json"))
        and os.path.isfile(
            os.path.join(mirror, "transformer", "diffusion_pytorch_model.safetensors.index.json")
        )
        and os.path.isfile(os.path.join(mirror, "vae", "diffusion_pytorch_model.safetensors"))
        and os.path.isfile(os.path.join(mirror, "scheduler", "scheduler_config.json"))
        and all(os.path.isdir(os.path.join(mirror, sub)) for sub in
                ("text_encoder", "tokenizer", "processor"))
    )


def _convert_transformer(raw_root: str, mirror: str) -> None:
    out_dir = os.path.join(mirror, "transformer")
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "config.json"), TRANSFORMER_CONFIG)

    src = os.path.join(raw_root, "transformer", "transformer.pth")
    logger.info("JoyAI mirror: converting transformer %s (this runs once).", src)
    sd = _load_raw_state_dict(src)
    weight_map: dict[str, str] = {}
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    shard_index = 1
    total_bytes = 0
    temp_names: dict[int, str] = {}
    for name, tensor in sd.items():
        name = _remap_transformer_key(name)
        nbytes = tensor.numel() * tensor.element_size()
        total_bytes += nbytes
        if current and current_bytes + nbytes > _TRANSFORMER_SHARD_BYTES:
            temp_name = f"joyai_tmp_shard-{shard_index:05d}.safetensors"
            save_file(current, os.path.join(out_dir, temp_name))
            for key in current:
                weight_map[key] = temp_name
            temp_names[shard_index] = temp_name
            logger.info("JoyAI mirror: transformer shard %d written (%.2f GiB).",
                        shard_index, current_bytes / 1024**3)
            current = {}
            current_bytes = 0
            shard_index += 1
        current[name] = tensor
        current_bytes += nbytes
    if current:
        temp_name = f"joyai_tmp_shard-{shard_index:05d}.safetensors"
        save_file(current, os.path.join(out_dir, temp_name))
        for key in current:
            weight_map[key] = temp_name
        temp_names[shard_index] = temp_name
        logger.info("JoyAI mirror: transformer shard %d written (%.2f GiB).",
                    shard_index, current_bytes / 1024**3)

    n_shards = shard_index
    final_names: dict[str, str] = {}
    for i in range(1, n_shards + 1):
        old = temp_names[i]
        new = f"diffusion_pytorch_model-{i:05d}-of-{n_shards:05d}.safetensors"
        os.replace(os.path.join(out_dir, old), os.path.join(out_dir, new))
        final_names[old] = new
    weight_map = {key: final_names[name] for key, name in weight_map.items()}
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    _write_json(
        os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"),
        index,
    )
    logger.info("JoyAI mirror: transformer conversion done (%d weights, %d shards).",
                len(weight_map), n_shards)


def _convert_vae(raw_root: str, mirror: str) -> None:
    out_dir = os.path.join(mirror, "vae")
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "config.json"), VAE_CONFIG)

    src = os.path.join(raw_root, "vae", "Wan2.1_VAE.pth")
    logger.info("JoyAI mirror: converting VAE %s.", src)
    sd = _load_raw_state_dict(src)
    converted: dict[str, torch.Tensor] = {}
    for raw_key, tensor in sd.items():
        target_key = map_vae_key(raw_key)
        if target_key in converted:
            raise KeyError(f"duplicate target key {target_key}")
        converted[target_key] = tensor.to(torch.bfloat16)

    from diffusers.models.autoencoders import AutoencoderKLWan

    model = AutoencoderKLWan()
    expected = set(dict(model.named_parameters()).keys())
    if set(converted) != expected:
        raise ValueError(
            f"VAE key mismatch after conversion: "
            f"{len(expected - set(converted))} missing / "
            f"{len(set(converted) - expected)} extra."
        )
    for name, param in model.named_parameters():
        if tuple(param.shape) != tuple(converted[name].shape):
            raise ValueError(f"VAE shape mismatch for {name}.")
    save_file(converted, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"))
    logger.info("JoyAI mirror: VAE conversion done (%d params).", len(converted))


def _write_static_files(raw_root: str, mirror: str) -> None:
    os.makedirs(os.path.join(mirror, "scheduler"), exist_ok=True)
    _write_json(os.path.join(mirror, "scheduler", "scheduler_config.json"), SCHEDULER_CONFIG)
    _write_json(os.path.join(mirror, "model_index.json"), MODEL_INDEX)

    te_source = os.path.join(raw_root, "JoyAI-Image-Und")
    for sub in ("text_encoder", "tokenizer", "processor"):
        link = os.path.join(mirror, sub)
        if os.path.islink(link) or os.path.isdir(link):
            if os.path.realpath(link) != os.path.realpath(te_source):
                logger.warning(
                    "JoyAI mirror %s already exists and points at %s; leaving it.",
                    link,
                    os.path.realpath(link),
                )
            continue
        os.symlink(te_source, link)
    logger.info("JoyAI mirror: text_encoder/tokenizer/processor linked to %s.", te_source)


def ensure_mirror(raw_root: str, mirror_dir: str | None = None, force: bool = False) -> str:
    """Idempotently build the Diffusers mirror for a raw JoyAI checkpoint."""
    mirror = mirror_dir or default_mirror_dir(raw_root)
    os.makedirs(mirror, exist_ok=True)
    if _mirror_complete(mirror) and not force:
        return mirror
    with _mirror_lock(raw_root):
        if _mirror_complete(mirror) and not force:
            return mirror
        tmp_mirror = tempfile.mkdtemp(prefix=".joyai-mirror-", dir=mirror)
        try:
            _convert_transformer(raw_root, tmp_mirror)
            _convert_vae(raw_root, tmp_mirror)
            _write_static_files(raw_root, tmp_mirror)
            for item in os.listdir(tmp_mirror):
                os.replace(
                    os.path.join(tmp_mirror, item),
                    os.path.join(mirror, item),
                )
            logger.info("JoyAI mirror ready at %s.", mirror)
        finally:
            with contextlib.suppress(OSError):
                os.rmdir(tmp_mirror)
    return mirror
