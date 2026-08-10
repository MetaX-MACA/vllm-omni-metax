#!/usr/bin/env python3
"""Generate the small deterministic assets used by the MiniMax-H3 tests.

Outputs (into OUT_DIR):
    first_frame.png  768x768 snowy-forest scene with a fox
    last_frame.png   768x768 warmer variant of the same scene
    ref_audio.wav    3 s / 32 kHz mono two-tone melody
"""

import math
import os
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw


def make_scene(path: str, *, fox_x: int, warm: bool) -> None:
    w = h = 768
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            t = y / h
            if warm:
                r, g, b = int(80 + 120 * (1 - t)), int(60 + 60 * t), int(110 + 90 * t)
            else:
                r, g, b = int(30 + 50 * (1 - t)), int(45 + 55 * (1 - t)), int(110 + 100 * t)
            px[x, y] = (r, g, b)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 560, w, h], fill=(225, 228, 240))
    for tx in (120, 260, 520, 640, 90, 700):
        base_y = 560
        d.polygon([(tx - 55, base_y), (tx, base_y - 150), (tx + 55, base_y)], fill=(30, 55, 90))
        d.polygon([(tx - 38, base_y - 40), (tx, base_y - 170), (tx + 38, base_y - 40)], fill=(36, 65, 105))
    d.ellipse([560, 70, 640, 150], fill=(245, 230, 170) if warm else (220, 225, 240))
    fx, fy = fox_x, 600
    d.ellipse([fx - 70, fy - 45, fx + 70, fy + 45], fill=(225, 120, 50))
    d.polygon([(fx - 70, fy - 20), (fx - 130, fy - 95), (fx - 55, fy - 40)], fill=(225, 120, 50))
    d.polygon([(fx + 70, fy - 20), (fx + 130, fy - 95), (fx + 55, fy - 40)], fill=(225, 120, 50))
    d.polygon([(fx - 125, fy - 90), (fx - 105, fy - 105), (fx - 95, fy - 85)], fill=(255, 250, 240))
    d.polygon([(fx + 105, fy - 90), (fx + 125, fy - 105), (fx + 95, fy - 85)], fill=(255, 250, 240))
    d.ellipse([fx - 28, fy - 30, fx - 8, fy - 12], fill=(40, 30, 25))
    d.ellipse([fx + 8, fy - 30, fx + 28, fy - 12], fill=(40, 30, 25))
    d.ellipse([fx - 10, fy + 8, fx + 10, fy + 28], fill=(50, 40, 35))
    for i in range(40):
        x = (i * 97) % w
        y = (i * 61) % 540
        d.ellipse([x, y, x + 4, y + 4], fill=(250, 252, 255))
    img.save(path)


def make_audio(path: str, seconds: float = 3.0, rate: int = 32000) -> None:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    freq = 440 * (1 + 0.3 * np.sin(2 * math.pi * 0.5 * t))
    amp = 0.35 * np.minimum(1, np.minimum(t / 0.1, (seconds - t) / 0.2))
    sig = amp * (np.sin(2 * math.pi * freq * t) + 0.4 * np.sin(2 * math.pi * 660 * t))
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3_ASSETS_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    make_scene(os.path.join(out_dir, "first_frame.png"), fox_x=384, warm=False)
    make_scene(os.path.join(out_dir, "last_frame.png"), fox_x=500, warm=True)
    make_audio(os.path.join(out_dir, "ref_audio.wav"))
    print(f"assets written to {out_dir}")


if __name__ == "__main__":
    main()
