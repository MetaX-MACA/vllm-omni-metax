# Reconstruction Notes

This page records a practical validation path for reproducing a working
`vllm-omni-metax` environment from scratch.

## Goal

Validate that:

- `vllm-metax` provides a healthy MetaX backend
- `vllm-omni` can discover an external platform plugin
- the `0.26.0` runtime patch layer is active when needed
- `vllm-omni-metax` activates and routes stage device visibility correctly

## Recommended reconstruction flow

1. Start from a clean Linux environment with MACA installed.
2. Verify raw device visibility through the MetaX runtime.
3. Install and validate `vllm-metax`.
4. Install `vllm-omni`.
5. Install `vllm-omni-metax` in editable mode.
6. Verify plugin activation with a one-line Python probe.
7. Run a small `Qwen3-Omni` or `Qwen-Image-Edit-2511` workload.
8. Move to larger multimodal pipelines only after the model-specific path is
   stable.

## Minimal checks

### Check backend runtime

```bash
python -c "from vllm_metax.utils import import_pymxsml; m=import_pymxsml(); m.nvmlInit(); print(m.nvmlDeviceGetCount()); m.nvmlShutdown()"
```

### Check plugin activation

```bash
python -c "from vllm_omni_metax.plugin import metax_omni_platform_plugin; print(metax_omni_platform_plugin())"
```

### Check forced activation path

```bash
export VLLM_OMNI_METAX_FORCE=1
python -c "from vllm_omni_metax.plugin import metax_omni_platform_plugin; print(metax_omni_platform_plugin())"
```

### Compare with patches disabled

```bash
export VLLM_OMNI_METAX_DISABLE_PATCHES=1
python -c "from vllm_omni_metax.plugin import metax_omni_platform_plugin; print(metax_omni_platform_plugin())"
```

## Common failure points

- `vllm-metax` is installed but its runtime libraries are not visible in
  `LD_LIBRARY_PATH`.
- The Python environment contains mismatched `vllm-omni` and
  `vllm-omni-metax` versions.
- `Qwen3-Omni` or `Qwen-Image-Edit-2511` reaches a rotary code path that still
  assumes CUDA-only flash-attn imports.
- Device visibility is filtered unexpectedly by container runtime settings.
- Debugging uses force-enable and then forgets to return to the default
  auto-detection path.

## Expected steady state

In a stable environment:

- MetaX devices are visible through `pymxsml`.
- The plugin resolves to `MetaxOmniPlatform` without force mode.
- The runtime patch layer is available for the affected Omni rotary path.
- Omni workers use the standard GPU worker path.
- Stage device control remains aligned with `MACA_VISIBLE_DEVICES`.
