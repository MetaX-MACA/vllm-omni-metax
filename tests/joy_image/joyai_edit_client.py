#!/usr/bin/env python3
"""Send an image-edit request to the JoyAI-Image-Edit vllm-omni server.

The client POSTs a standard OpenAI chat request with one text instruction and
one reference image, then decodes the returned image to a PNG file.  When the
input image path does not exist it generates a simple red-circle test image so
the smoke test is self-contained.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


def _default_test_image(path: str, size: int = 1024) -> None:
    img = Image.new("RGB", (size, size), (210, 210, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size // 12, size // 12, size - size // 12, size - size // 12],
                   fill=(160, 160, 160), outline=(90, 90, 90), width=12)
    draw.ellipse([size // 3, size // 3, size * 2 // 3, size * 2 // 3],
                 fill=(220, 60, 60), outline=(120, 20, 20), width=10)
    img.save(path)
    print(f"generated test input image: {path}", flush=True)


def _decode_response_image(result: dict) -> str:
    content = result["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = part["image_url"]["url"]
            return url.split(",", 1)[1] if "," in url else url
    raise RuntimeError(f"no image_url part in response: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",
                        default="http://127.0.0.1:8094/v1/chat/completions")
    parser.add_argument("--model", required=True,
                        help="served model id (--served-model-name, e.g. joyai-edit)")
    parser.add_argument("--image", default="")
    parser.add_argument("--prompt", default="Turn the red circle blue")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true",
                        help="basic sanity check of the decoded output PNG")
    args = parser.parse_args()

    image_path = args.image
    if not image_path or not os.path.isfile(image_path):
        image_path = image_path or "/tmp/joyai_test_input.png"
        _default_test_image(image_path)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "extra_body": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "true_cfg_scale": args.cfg_scale,
            "seed": args.seed,
        },
    }
    request = urllib.request.Request(
        args.api_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"POST {args.api_url} prompt={args.prompt!r} steps={args.steps} "
          f"cfg={args.cfg_scale} size={args.width}x{args.height}", flush=True)
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.loads(response.read())

    image_data = base64.b64decode(_decode_response_image(result))
    with open(args.output, "wb") as f:
        f.write(image_data)
    print(f"saved edited image: {args.output} ({len(image_data)} bytes)", flush=True)

    if args.verify:
        image = Image.open(args.output).convert("RGB")
        print(f"output size: {image.size}", flush=True)
        if tuple(image.size) != (args.width, args.height):
            raise SystemExit(
                f"unexpected output size {image.size}, expected {args.width}x{args.height}"
            )
        array = np.asarray(image, dtype=np.float32)
        if array.std() < 10:
            raise SystemExit("output looks flat/blank; edit likely failed")
        print("output sanity check OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # keep server logs readable on failure
        print(f"joyai_edit_client failed: {exc}", file=sys.stderr)
        sys.exit(1)
