#!/usr/bin/env bash
# Install the system dependencies required by the MiniMax-H3 tests.
#
# Ref2VA video-reference preprocessing shells out to `ffmpeg` and `ffprobe`.
# MetaX images ship ffmpeg under /opt/maca-*/ffmpeg/bin but:
#   1. ffprobe is not on PATH;
#   2. ffmpeg misses libxcb-shm0/libxcb-shape0/libxcb-xfixes0.
set -euo pipefail

# 1) libxcb libraries required by the bundled ffmpeg
if command -v apt-get >/dev/null 2>&1; then
    apt-get install -y libxcb-shm0 libxcb-shape0 libxcb-xfixes0
fi

# 2) make ffprobe/ffmpeg available on PATH
if command -v ffprobe >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg/ffprobe already on PATH"
    exit 0
fi

FF_BIN=""
for d in /opt/maca-*/ffmpeg/bin /opt/maca/ffmpeg/bin; do
    if [[ -x "${d}/ffmpeg" && -x "${d}/ffprobe" ]]; then
        FF_BIN="${d}"
        break
    fi
done

if [[ -n "${FF_BIN}" ]]; then
    ln -sf "${FF_BIN}/ffmpeg" /usr/local/bin/ffmpeg
    ln -sf "${FF_BIN}/ffprobe" /usr/local/bin/ffprobe
    echo "linked ffmpeg/ffprobe from ${FF_BIN}"
else
    echo "no bundled ffmpeg found; trying apt-get install ffmpeg"
    apt-get install -y ffmpeg
fi

ffmpeg -version | head -1
ffprobe -version | head -1
