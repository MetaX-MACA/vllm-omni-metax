#!/bin/bash

# CustomVoice model requires task_type=CustomVoice + a built-in speaker;
# task_type=Base (ref_audio voice cloning) is not supported by this variant.
python openai_speech_client.py --model /mxstorage/pde_ai/models/llm/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/ \
    --task-type CustomVoice \
    --speaker vivian \
    --text "hello,this is a test voice" \
    --api-base http://127.0.0.1:8500 \
    --output tts_output.wav
