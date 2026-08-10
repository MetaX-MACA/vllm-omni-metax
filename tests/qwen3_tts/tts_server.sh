#!/bin/bash

# Qwen3-TTS e2e server (vllm-omni 0.26).
# The plugin deploy patch resolves the MetaX-tuned 2-GPU layout
# (src/vllm_omni_metax/deploy/qwen3_tts.yaml) automatically; pass
# --deploy-config to override, e.g.:
#   --deploy-config src/vllm_omni_metax/deploy/qwen3_tts.yaml
# Collapse both stages to one GPU by setting devices: "0" in the YAML.
# --enforce-eager is intentionally NOT passed: the maca platform section in
# qwen3_tts.yaml runs the code2wav stage eager (MetaX driver cannot create
# streams on a device with active CUDA graphs) while the talker keeps CUDA
# graphs.

vllm serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-Base/ \
    --omni \
    --trust-remote-code
