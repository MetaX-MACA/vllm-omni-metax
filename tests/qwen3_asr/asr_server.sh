#!/bin/bash

# pip install soundfile librosa
# pip install strenum
# pip uninstall xgrammar -y && pip install xgrammar==0.1.33
# #fix /opt/conda/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py +55
# pip install av

vllm serve /external/ai/models/llm/Qwen/Qwen3-ASR-1.7B/ --served-model-name qwen3-asr-1.7b --disable-log-stats