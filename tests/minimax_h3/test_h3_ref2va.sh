#!/usr/bin/env bash
# Ref2VA tests (Ref2VA partition, port 8092):
#   1) image + audio reference   -> video with lip/audio conditioning
#   2) video reference           -> video continuation reusing its soundtrack
#
# Env overrides:
#   API_URL         server endpoint (default http://127.0.0.1:8092/v1/videos/sync)
#   OUT_DIR         output dir      (default /tmp/h3_tests)
#   H3_ASSETS_DIR   dir with first_frame.png + ref_audio.wav
#   NUM_INFERENCE_STEPS / H3_DURATION
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="${API_URL:-http://127.0.0.1:8092/v1/videos/sync}"
OUT_DIR="${OUT_DIR:-/tmp/h3_tests}"
ASSETS="${H3_ASSETS_DIR:-${HERE}/assets}"
STEPS="${NUM_INFERENCE_STEPS:-20}"
DURATION="${H3_DURATION:-4.0}"
REF_IMAGE="${ASSETS}/first_frame.png"
REF_AUDIO="${ASSETS}/ref_audio.wav"

for f in "${REF_IMAGE}" "${REF_AUDIO}"; do
    if [[ ! -f "${f}" ]]; then
        echo "error: ${f} not found; run make_assets.py first" >&2
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

# The base64 data URL can exceed the shell argv limit, so write it to a file
# and let curl read the field value from the file (-F 'name=<file').
AUDIO_B64="$(base64 -w0 "${REF_AUDIO}")"
printf '{"audio_url":"data:audio/wav;base64,%s"}' "${AUDIO_B64}" > "${OUT_DIR}/audio_ref.json"

echo "== Ref2VA image+audio =="
curl -sS -X POST "${API_URL}" \
    -F 'prompt=A white cat with black mustache markings sits on a snowy porch, lip-syncing softly to the reference melody before looking toward the camera; a gentle breeze carries the tune and snow crunches nearby.' \
    -F width=1024 \
    -F height=576 \
    -F fps=24 \
    -F "num_inference_steps=${STEPS}" \
    -F flow_shift=12 \
    -F seed=3101 \
    -F "extra_params={\"task\":\"ref2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
    -F "input_reference=@${REF_IMAGE};type=image/png" \
    -F "audio_reference=<${OUT_DIR}/audio_ref.json" \
    -o "${OUT_DIR}/ref2va_image_audio.mp4" \
    -w 'HTTP %{http_code} time %{time_total}s size %{size_download}\n'
echo "Ref2VA image+audio output: ${OUT_DIR}/ref2va_image_audio.mp4"

echo "== Ref2VA video reference =="
# `input_references` (plural) is persisted by the server as file paths, which
# is what the H3 video-reference preprocessing requires.  Using the singular
# `input_reference` decodes the upload to PIL frames and fails with
# "multi-video Ref2VA currently requires file paths".
VIDEO_REF="${VIDEO_REF:-${OUT_DIR}/fl2va_first_frame.mp4}"
if [[ ! -f "${VIDEO_REF}" ]]; then
    echo "warning: reference video ${VIDEO_REF} missing; skipping video-reference case" >&2
    exit 0
fi

curl -sS -X POST "${API_URL}" \
    -F 'prompt=The fox from the reference video continues walking through the snowy forest; the same forest ambience and footsteps continue, and the camera slowly tilts up toward the glowing sky.' \
    -F width=1024 \
    -F height=576 \
    -F fps=24 \
    -F "num_inference_steps=${STEPS}" \
    -F flow_shift=12 \
    -F seed=3201 \
    -F "extra_params={\"task\":\"ref2va\",\"duration\":${DURATION},\"audio_flow_shift\":3.0}" \
    -F "input_references=@${VIDEO_REF};type=video/mp4" \
    -o "${OUT_DIR}/ref2va_video_ref.mp4" \
    -w 'HTTP %{http_code} time %{time_total}s size %{size_download}\n'
echo "Ref2VA video-reference output: ${OUT_DIR}/ref2va_video_ref.mp4"
