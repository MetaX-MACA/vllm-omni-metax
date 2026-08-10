# Quickstart

The shortest path is to start from a working `vllm-metax` environment and then
layer `vllm-omni` and `vllm-omni-metax` into it.

## Recommended deployment path

For most users, the recommended bring-up path is:

1. Start from a MetaX image or host where `vllm-metax` already works.
2. Install `vllm-omni`.
3. Install `vllm-omni-metax`.
4. Launch your normal `vllm-omni` workflow.

This mirrors the guidance used by `vLLM-metax`: get the hardware backend stable
first, then add higher-level functionality.


## Releases

*Below is version mapping to released plugin and `mcoplib` with `maca`:*

| plugin version | maca version | mcoplib version | docker image url |
|:--------------:|:------------:|:---------------:|:----------------:|
| `v0.18.0` | `maca3.5.3.x` | `0.4.3` | [vllm-metax:0.18.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.18.0-torch2.8) |
| `v0.19.0` | `maca3.5.3.x` | `0.4.4` | [vllm-metax:0.19.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.19.0-torch2.8) |
| `v0.20.0` | `maca3.5.3.x` | `0.4.5` | -- |
| `v0.22.0` | `maca3.8.0.x` | `0.4.7` | -- |
| `v0.26.0` | `maca3.8.2.x` (placeholder — confirm with the vllm-metax release table) | `0.4.x` | -- |

!!! warning "Usage Warning"
    **vLLM-Omni-MetaX is intended to work out of the box with the matching Docker images listed above.**

    All VLM tests are based on the related `maca` version. Using an incompatible version of `maca` for `vllm-omni-metax` may cause unexpected bugs or errors. This is not guaranteed.


## What changes after installation

You do not need a separate user-facing launcher from this repository.
Once the plugin is installed:

- `vllm-omni` keeps its normal CLI and workflow.
- The adapter activates only when MetaX runtime is detected, unless forced.
- A runtime compatibility patch may be applied during activation for the Omni
  rotary path.
- Omni AR and generation workers continue to use GPU worker classes.
- Diffusion attention backend selection follows a conservative policy:
  `FLASH_ATTN` when both capability and package checks pass, otherwise
  `TORCH_SDPA`.

## Next steps

- Follow [Installation](installation.md) for the full environment setup.
- Read [User Guide](../user_guide/overview.md) for activation and behavior
  details.
- Read [Architecture](../developer_guide/architecture.md) if you plan to extend
  the plugin.
