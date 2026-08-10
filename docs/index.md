# vLLM-Omni MetaX

![vLLM x MetaX](assets/logos/vllm-metax-logo.webp){ .hero-banner }

`vllm-omni-metax` is the MetaX adapter plugin for
[`vllm-omni`](https://github.com/vllm-project/vllm-omni). It reuses the
existing [`vllm-metax`](https://github.com/MetaX-MACA/vLLM-metax) hardware
backend and adds the glue required for Omni's multi-stage multimodal runtime.

In practice, the responsibilities are split as follows:

| Component | Responsibility |
| --- | --- |
| `vllm-metax` | MetaX hardware platform, kernels, runtime integration |
| `vllm-omni` | Multi-stage multimodal inference pipeline |
| `vllm-omni-metax` | Omni platform plugin that bridges the two |

## Why this project exists

`vllm-omni` ships with GPU-oriented execution paths, but MetaX enablement should
stay owned by the MetaX backend instead of being maintained as an in-tree fork.
This repository keeps that boundary clear:

- It does not reimplement a new hardware backend.
- It detects whether a usable MetaX runtime is present.
- It registers an out-of-tree Omni platform plugin only when activation is
  appropriate.
- It applies a focused runtime patch only where upstream CUDA-only assumptions
  block MetaX execution in `vllm-omni 0.26.0`.
- It mirrors Omni's stage device selection to the MetaX runtime environment.

## Recommended workflow

If you are deploying on MetaX hardware, the recommended order is:

1. Prepare a working `vllm-metax` environment.
2. Install `vllm-omni`.
3. Install `vllm-omni-metax`.
4. Start the usual `vllm-omni` workflow and let this plugin activate
   automatically.

See:

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [User Guide](user_guide/overview.md)
- [Architecture](developer_guide/architecture.md)

## Design goals

- Reuse the existing MetaX platform implementation from `vllm-metax`.
- Keep compatibility fixes small, explicit, and runtime-scoped instead of
  carrying a source fork of `vllm-omni`.
- Keep operational behavior predictable for users already familiar with
  `vllm-metax`.
- Make debugging explicit with environment-controlled enable and disable
  switches.
