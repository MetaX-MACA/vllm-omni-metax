#!/usr/bin/env python3
"""Verify JoyAI-Image-Und (Qwen3-VL) served through vllm-omni --omni.

Checks one text-only chat and one image+text chat, asserting on the decoded
answers so a broken pipeline fails loudly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request

from PIL import Image, ImageDraw


def _chat(api_url: str, model: str, messages: list[dict], max_tokens: int = 64) -> str:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    return result["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",
                        default="http://127.0.0.1:8095/v1/chat/completions")
    parser.add_argument("--model", required=True,
                        help="served model id (--served-model-name, e.g. joyai-und)")
    parser.add_argument("--image", default="",
                        help="image for the vision check (default: generated)")
    args = parser.parse_args()

    # 1. Text-only chat.
    answer = _chat(
        args.api_url,
        args.model,
        [{"role": "user", "content": "Reply with exactly: hello from qwen3vl"}],
    )
    print(f"text answer: {answer!r}", flush=True)
    if "hello from qwen3vl" not in answer.lower():
        raise SystemExit(f"unexpected text answer: {answer!r}")

    # 2. Vision chat: blue circle on gray background.
    image_path = args.image
    if not image_path or not os.path.isfile(image_path):
        image_path = image_path or "/tmp/joy_und_vision.png"
        img = Image.new("RGB", (256, 256), (200, 200, 200))
        ImageDraw.Draw(img).ellipse([64, 64, 192, 192], fill=(0, 40, 220))
        img.save(image_path)
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    answer = _chat(
        args.api_url,
        args.model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the circle? One word."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    print(f"vision answer: {answer!r}", flush=True)
    if "blue" not in answer.lower():
        raise SystemExit(f"unexpected vision answer: {answer!r}")

    print("JoyAI-Image-Und (Qwen3-VL) test OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"joy_und_client failed: {exc}", file=sys.stderr)
        sys.exit(1)
