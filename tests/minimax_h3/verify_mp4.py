#!/usr/bin/env python3
"""Verify that one or more MP4 files contain both a video and an audio track.

Dependency-free: walks the MP4 box tree looking for `hdlr` handlers.
Prints `vide`/`soun` per file and exits non-zero when either track is missing.
"""

import struct
import sys


def _iter_boxes(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size = struct.unpack(">I", buf[i : i + 4])[0]
        typ = buf[i + 4 : i + 8].decode("latin1")
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8 : i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
            hdr = 8
        else:
            hdr = 8
        if size < hdr:
            break
        yield typ, i, i + size
        i += size


def handlers(path: str) -> list[str]:
    with open(path, "rb") as f:
        data = f.read()
    found: list[str] = []

    def walk(start: int, end: int) -> None:
        for typ, a, b in _iter_boxes(data, start, end):
            if typ in ("moov", "trak", "mdia", "minf", "stbl"):
                walk(a + 8, b)
            elif typ == "hdlr" and b - a >= 20:
                found.append(data[a + 16 : a + 20].decode("latin1"))

    walk(0, len(data))
    return found


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <file.mp4> [...]")
        return 2
    failed = False
    for path in sys.argv[1:]:
        hs = handlers(path)
        ok = "vide" in hs and "soun" in hs
        print(f"{'PASS' if ok else 'FAIL'} {path} -> {hs}")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
