# JoyAI-Image-Edit model test scripts

Model-level smoke tests for the JoyAI pipeline shipped via the metax plugin
(upstream vllm-omni PR #4112 backport + raw-checkpoint adapter).  They follow
the same server + client pattern as `tests/wan2_2` / `tests/qwen3_tts` and run
real inference against the models.

## 1. Full pipeline: JoyAI-Image-Edit (image edit)

Terminal 1 - start the server (raw checkpoint is served directly; the plugin
builds the Diffusers mirror on first use):

```bash
./tests/joy_image/run_joyai_edit_server.sh
```

Terminal 2 - send an edit request and verify the output image:

```bash
./tests/joy_image/test_joyai_edit.sh
```

The client POSTs "Turn the red circle blue" with a generated red-circle input
image, decodes the edited 1024x1024 PNG and sanity-checks its size/statistics.

Env overrides: `MODEL`, `PORT`, `MIRROR_DIR`, `TENSOR_PARALLEL`,
`VAE_USE_TILING` (server); `API_URL`, `IMAGE`, `OUT`, `PROMPT`,
`HEIGHT`, `WIDTH`, `NUM_INFERENCE_STEPS`, `CFG_SCALE`, `SEED` (client).
The server registers the model as `joyai-edit` (`--served-model-name`); the
client defaults to that name and `MODEL` must match the served name.

## 2. Text encoder: JoyAI-Image-Und (Qwen3-VL, --omni)

Terminal 1:

```bash
./tests/joy_image/run_joy_und_server.sh
```

Terminal 2:

```bash
./tests/joy_image/test_joy_und.sh
```

The client verifies one text-only chat and one image+text vision chat.
The `qwen3_vl` single-LLM-stage omni pipeline is registered by the plugin, so
the raw `JoyAI-Image-Und` checkpoint serves with `--omni` directly.

Env overrides: `MODEL`, `PORT`, `TP`, `GPU_MEM` (server); `API_URL`, `IMAGE`
(client). The server registers the model as `joyai-und`; the client defaults
to that name.

## Notes

- Requires a MetaX container with `vllm-omni` / `vllm-omni-metax` installed
  and the raw checkpoints under `/mxstorage/pde_ai/models/llm/Diffusion-models/`.
- The edit pipeline is validated on a single MXC500 (64 GB) with VAE
  tiling/slicing enabled (decode otherwise hits CUDNN OOM).
