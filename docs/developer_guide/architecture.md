# Architecture

`vllm-omni-metax` is intentionally implemented as a thin adapter instead of a
forked backend.

## High-level structure

The repository currently has three core pieces:

| File | Responsibility |
| --- | --- |
| `plugin.py` | Runtime detection, patch application, and plugin activation |
| `platform.py` | Omni platform class built on top of `MacaPlatform` |
| `patches/rope_patch.py` | Runtime shim for Omni rotary import compatibility |

## Entry-point layer

`plugin.py` exposes the entry point consumed by `vllm-omni`:

```python
metax = "vllm_omni_metax.plugin:metax_omni_platform_plugin"
```

The function is intentionally narrow in scope:

- It honors explicit disable and force environment flags.
- It probes runtime availability through `vllm_metax.utils.import_pymxsml()`.
- It can apply a focused runtime patch before handing control to Omni.
- It returns a class path only when the MetaX runtime looks usable.

This keeps the plugin safe to install in mixed environments where MetaX runtime
may or may not be present.

## Patch layer

`patches/rope_patch.py` exists because parts of `vllm-omni 0.26.0` still assume
CUDA-oriented flash-attn rotary imports. On MetaX, that assumption can break
model startup even though the rest of the stack is otherwise usable.

The patch layer therefore:

- installs a shim for `vllm.vllm_flash_attn.layers.rotary`
- routes rotary application through either external flash-attn or a torch
  fallback
- targets the model paths exercised during `Qwen3-Omni` and
  `Qwen-Image-Edit-2511` bring-up

This is a runtime compatibility patch, not a forked copy of Omni model logic.

## Platform layer

`MetaxOmniPlatform` combines Omni semantics with MetaX backend behavior:

- `OmniPlatform` provides the interface expected by `vllm-omni`.
- `MacaPlatform` provides the concrete MetaX implementation.

Key design choices:

- Omni currently has no dedicated MetaX enum, so the plugin maps itself to the
  CUDA-like Omni platform enum.
- Worker classes remain the standard GPU-oriented Omni workers.
- Device capability, device names, and free-memory queries are handled through
  torch CUDA-shaped APIs exposed by the MetaX stack.
- Device count still comes from the underlying `MacaPlatform`.
- Device control environment variables are mirrored to
  `CUDA_VISIBLE_DEVICES` and `MACA_VISIBLE_DEVICES`.

## Why the plugin reports CUDA-like behavior

`vllm-omni` currently expects GPU-oriented behaviors through CUDA-shaped APIs.
MetaX already provides a CUDA-alike execution experience through the
`vllm-metax` stack, so this plugin adapts to Omni's expectation rather than
introducing a new incompatible execution model.

That is why methods such as `get_torch_device()` and `synchronize()` still use
`torch.cuda` style interfaces.

## Attention backend selection

The diffusion attention backend policy has two goals:

1. Stay close to upstream Omni behavior.
2. Avoid selecting `FLASH_ATTN` unless both package and capability checks pass.

When `FLASH_ATTN` is unavailable, the plugin falls back to `TORCH_SDPA` with a
warning instead of failing hard.

## Extension guidance

If you extend this repository, keep the ownership boundary intact:

- Hardware-specific logic should stay in `vllm-metax` whenever possible.
- Omni integration logic should stay here.
- Keep compatibility patches as small runtime shims instead of carrying a large
  source fork.

In general, the adapter should remain small, explicit, and easy to diff against
upstream changes.
