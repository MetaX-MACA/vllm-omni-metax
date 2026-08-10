#!/bin/bash

# NEW 0.26-era verification: Qwen3-Omni-MoE thinking variant using the
# builtin vllm_omni/deploy/qwen3_omni_moe_thinking.yaml.
# Assert: multimodal chat returns a thinking/reflection phase before the
# final answer.
#
# NOTE: devices/topology in the builtin file are single-GPU per stage; on a
# MACA box use the plugin qwen3_omni_moe.yaml layout (thinker TP=2 on
# "0,1", talker/code2wav on "2") and add the thinking settings from the
# builtin variant if the thinker stage does not fit one GPU.

vllm serve /external/ai/models/llm/Qwen/Qwen3-Omni-30B-A3B-Instruct/ \
    --omni \
    --trust-remote-code \
    --enforce-eager \
    --deploy-config qwen3_omni_moe_thinking.yaml
