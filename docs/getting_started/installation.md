# Installation

This page follows the same high-level approach as `vLLM-metax`: prepare a valid
MetaX runtime first, then install the Omni-side adapter on top.

## Requirements

- OS: Linux
- Python: 3.10 -- 3.12
- Hardware: MetaX C-series GPU
- SDK: MACA runtime correctly installed
- System tools: `ffmpeg` / `ffprobe` on `PATH` (required by diffusion-model
  video-reference preprocessing, e.g. MiniMax-H3 Ref2VA)
- Base software:
  - `vllm-metax`
  - `vllm-omni`

## Compatibility notes

`vllm-omni-metax` is a thin Python plugin. It does not ship extra kernels by
itself and therefore depends on a healthy `vllm-metax` installation. In the
`0.26.0` release line it ships runtime patches for:

- the Omni rotary embedding import path that otherwise assumes CUDA-only
  flash-attn bindings (`vllm.vllm_flash_attn.layers.rotary` shim)
- the Qwen3-Omni CodePredictor attention dtype (SDPA output cast to
  `o_proj.weight.dtype`)
- the Qwen3-TTS Code2Wav inner CUDA graph and Triton SnakeBeta (neutralized;
  opt-in via `VLLM_OMNI_METAX_ENABLE_*`)
- deploy YAML resolution (plugin `deploy/` directory takes precedence over
  builtin `vllm_omni/deploy/`)

Python-side runtime dependencies are listed in `requirements/common.txt`,
including `cache-dit==1.3.0`: `vllm-omni` 0.26 diffusion pipelines
(MiniMax-H3 among them) import `cache_dit`, but MetaX runtime wheels do not
pull it in, so it must be installed explicitly.

### System dependencies (ffmpeg / ffprobe)

Some diffusion pipelines (MiniMax-H3 Ref2VA video references, MP4 output)
shell out to `ffmpeg` and `ffprobe`. MetaX images usually bundle them under
`/opt/maca-<ver>/ffmpeg/bin`, but `ffprobe` is not on `PATH` and the bundled
`ffmpeg` misses the XCB libraries:

```bash
apt-get install -y libxcb-shm0 libxcb-shape0 libxcb-xfixes0
ln -sf /opt/maca-3.8.2/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg
ln -sf /opt/maca-3.8.2/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe
```

If no bundled ffmpeg exists, install it with `apt-get install -y ffmpeg`.
The MiniMax-H3 test suite ships a helper that performs both steps:
`tests/minimax_h3/install_system_deps.sh`.

For this repository revision:

- `vllm-omni-metax`: `0.26.0`
- `vllm-omni`: `0.26.0`
- `vllm_metax`: `0.26.0`

When in doubt, keep the three components aligned to the same release train used
by your MetaX environment or Docker image.

## Prepare the MetaX environment

If you already have a working `vllm-metax` container or host environment, you
can reuse it directly.

When building from source, the common MACA environment variables typically look
like this (this repository also ships them as `env.sh` — `source env.sh [/path/to/maca]`):

```bash
export MACA_PATH="/opt/maca"
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}"
```

For the detailed backend-side build flow, refer to the
[`vLLM-metax` installation document](https://vllm-metax.readthedocs.io/en/latest/getting_started/installation/maca.html).

## Install `vllm-metax`

`vllm-omni-metax` assumes the MetaX backend is already available. Install
`vllm-metax` first using your preferred method:

- MetaX released Docker image for the matching MACA version
- Editable source install for development
- Prebuilt wheel inside an existing MetaX software environment

## Install `vllm-omni`

Install `vllm-omni` in the same Python environment that already contains
`vllm-metax`. The `0.26.0` line is built from source on top of the matching
vLLM release (vllm-omni does not compile vLLM itself):

```bash
uv pip install vllm==0.26.0          # vLLM first; drives vllm-omni's deps
git clone https://github.com/vllm-project/vllm-omni.git -b v0.26.0
cd vllm-omni
uv pip install -e .
```

Install order matters: `vllm-metax` before `vllm-omni` (vllm-omni's setup
probes torch; vllm-metax patches vllm internals).

If your team already maintains a known-good `vllm-omni` checkout, use that
exact revision. Otherwise, keep it aligned with the version required by this
package.

## Install `vllm-omni-metax`

Clone this repository and install it into the same environment:

```bash
git clone https://github.com/MetaX-MACA/vllm-omni-metax.git
cd vllm-omni-metax
git checkout the/your/choice/branch
pip install -e .
```

If you use `uv`, the equivalent command is:

```bash
uv pip install -e .
```

Because this package is only an adapter layer, editable mode is the recommended
choice during bring-up and debugging.

