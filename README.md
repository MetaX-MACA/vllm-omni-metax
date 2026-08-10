# vllm-omni-metax

A thin adapter plugin that lets `vllm-omni` use the already-installed
`vllm-metax` platform implementation without modifying `vllm-omni` source files.

## About

vLLM-Omni MetaX is an adapter plugin that enables **vLLM-Omni** to run on
**MetaX GPUs** by leveraging the existing **vllm-metax backend**.

Unlike `vllm-metax`, which provides the low-level hardware platform
integration, this plugin focuses on bridging the **vLLM-Omni execution stack**
(multi-stage, multimodal pipelines) to the MetaX runtime.

## Responsibilities

| Component | Responsibility |
|----------|---------------|
| vllm-metax | Provides MetaX platform, kernels, and runtime |
| vllm-omni | Handles multi-stage multimodal inference |
| vllm-omni-metax | Bridges Omni to MetaX backend |

## Design rules

- `vllm-metax` remains the only owner of the MetaX/vLLM platform.
- `vllm-omni-metax` only provides an Omni platform plugin.
- No override of `vllm.platform_plugins`.
- No monkey-patching of `vllm-metax` registration.

## Prerequisites

- MetaX GPU (C-series)
- Linux
- Python >= 3.12
- `vllm` / `vllm-metax` / `vllm-omni` aligned to the same release train

## Installation

Install vllm-omni-metax as a Python package:

```bash
pip install -e .
```

## Compatibility

The current documentation and adaptation target the `0.26.0` stack:

- `vllm-omni-metax 0.26.0`
- `vllm-omni 0.26.0`
- `vllm-metax 0.26.0` (MACA 3.8.2.x placeholder — confirm against the
  vllm-metax release table before shipping)

See [docs/getting_started/installation.md](docs/getting_started/installation.md)
for the full setup flow and runtime patch notes.

## Verify

```bash
python - <<'PY'
from importlib.metadata import entry_points
print(entry_points(group="vllm_omni.platform_plugins"))
PY
```

## Optional environment controls

- `VLLM_OMNI_METAX_FORCE=1`: force-enable this plugin if MetaX detection is flaky.
- `VLLM_OMNI_METAX_DISABLE=1`: disable this plugin.
- `VLLM_OMNI_METAX_DISABLE_PATCHES=1`: skip the runtime monkey-patches.
- `VLLM_OMNI_METAX_ENABLE_CODE2WAV_CUDAGRAPH=1`: keep the Qwen3-TTS Code2Wav
  inner CUDA graph (default: neutralized as no-op).
- `VLLM_OMNI_METAX_ENABLE_TRITON_SNAKEBETA=1`: keep the Triton SnakeBeta kernel
  (default: forced eager).
