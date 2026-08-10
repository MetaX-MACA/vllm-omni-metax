# User Guide Overview

This project is intentionally small. Most user-facing behavior comes from the
interaction between `vllm-omni`, `vllm-metax`, and the runtime environment.

## How activation works

The plugin entry point is:

```python
vllm_omni_metax.plugin:metax_omni_platform_plugin
```

At startup, it follows this logic:

1. If `VLLM_OMNI_METAX_DISABLE` is set, the plugin stays disabled.
2. If `VLLM_OMNI_METAX_FORCE` is set, the plugin activates immediately.
3. Otherwise, it asks `vllm-metax` to probe the MetaX runtime through
   `pymxsml`.
4. During activation, it may install MetaX-specific runtime patches needed by
   `vllm-omni 0.26.0` model paths.
5. Only when the runtime probe succeeds does it register the Omni platform
   class.

This keeps runtime ownership with the MetaX backend while allowing a narrow
compatibility fix where upstream CUDA-only imports would otherwise block MetaX.

## Runtime patch layer

The new patch layer is intentionally narrow:

- It is enabled by default.
- It can be disabled with `VLLM_OMNI_METAX_DISABLE_PATCHES=1`.
- It installs a shim for the rotary embedding import path used by Omni
  diffusion/image code.
- Its current purpose is to support `Qwen3-Omni` and
  `Qwen-Image-Edit-2511` on the `0.26.0` stack.

## Runtime behavior

`MetaxOmniPlatform` inherits from both:

- `vllm_omni.platforms.interface.OmniPlatform`
- `vllm_metax.platform.MacaPlatform`

That means the plugin reuses the concrete MetaX hardware implementation instead
of creating a parallel backend.

## Device visibility

During Omni stage setup, the plugin keeps environment variables synchronized:

- `CUDA_VISIBLE_DEVICES`
- `MACA_VISIBLE_DEVICES`

For operators, this means stage-level worker placement is easier to reason
about, especially in multi-device debugging sessions.

## Attention backend policy

For diffusion attention, the plugin deliberately keeps a conservative policy:

- If the selected backend is explicitly requested and supported, it is used.
- If `FLASH_ATTN` is requested but capability or package checks fail, the plugin
  falls back to `TORCH_SDPA`.
- If no backend is specified, it prefers `FLASH_ATTN` only when both hardware
  capability and package availability checks pass.

This keeps behavior close to the upstream GPU-oriented policy while avoiding
aggressive assumptions on MetaX systems.

## Operational tips

- Start with automatic detection first and only use `VLLM_OMNI_METAX_FORCE=1`
  when isolating startup problems.
- Keep `VLLM_OMNI_METAX_DISABLE_PATCHES=1` for A/B debugging only, not as the
  default deployment mode for `0.26.0`.
- Treat `vllm-metax` as the source of truth for runtime health.
- Keep version combinations stable across `vllm-metax`, `vllm-omni`, and this
  repository.
- If Omni behavior changes after an upstream upgrade, re-verify platform plugin
  discovery before debugging model logic.
