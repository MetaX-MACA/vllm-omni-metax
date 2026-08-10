#!/bin/bash

# Qwen3-Omni-MoE e2e server (vllm-omni 0.26).
# The plugin deploy patch resolves the MetaX-tuned 3-GPU layout (thinker
# TP=2 on devices "0,1", talker/code2wav on "2") automatically.  Pass
# --deploy-config to override, e.g.:
#   --deploy-config src/vllm_omni_metax/deploy/qwen3_omni_moe.yaml
# --enforce-eager is intentionally NOT passed: the maca platform section in
# qwen3_omni_moe.yaml runs the code2wav stage eager (MetaX driver cannot
# create streams on a device with active CUDA graphs) while stages 0/1 keep
# CUDA graphs.

VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /mxstorage/pde_ai/models/llm/Qwen/Qwen3-Omni-30B-A3B-Instruct/ --omni
