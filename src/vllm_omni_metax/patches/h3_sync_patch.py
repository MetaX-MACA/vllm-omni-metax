# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MiniMax-H3 MetaX correctness fixes.

These are the minimal runtime fixes needed to keep MiniMax-H3 T2VA, FL2VA and
Ref2VA video output clean on the MetaX stack with vllm-omni 0.28:

1. ``MiniMaxH3Pipeline.forward`` trims the VAE alignment padding (17n+5 grid)
   back to the requested frame count; otherwise the MetaX VAE can decode the
   padding tail as noise and it is written into the muxed MP4.

2. ``MiniMaxH3Pipeline.forward`` forces a final device sync before returning
   the prepared uint8 video.  On MetaX the engine can otherwise read the
   worker's uint8 buffer before the async quantize/permute/trim operations
   land, which corrupts a later request on the same server (gray/noise video).

3. MiniMax-H3 keyframe and reference-video VAE encodes run rank-local.  The
   single-image FL2VA keyframe and the Ref2VA reference-video encode are
   sensitive to the 2-GPU tile parallel collective; isolating them keeps the
   reference latent deterministic and avoids all-gray video output.

The patch is applied lazily because the omni platform plugin activates while
vllm-omni is still initialising.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_PATCHED = False
_MAX_RETRIES = 600  # 1s interval; service startup can take minutes
_RETRY_COUNT = 0


def _wrap_forward_trim_and_sync(pipeline_cls: type) -> None:
    """Trim VAE padding tail and force a final GPU sync before return."""

    original_forward = pipeline_cls.forward

    def forward_trim_sync(self, request):
        from vllm_omni.platforms import current_omni_platform  # lazy import

        output = original_forward(self, request)
        if output is None or not isinstance(output.output, tuple):
            return output
        video, audio = output.output
        if video is None or not hasattr(video, "shape"):
            return output

        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        fps = float(getattr(sampling, "fps", None) or 24.0)
        duration = extra.get("duration_seconds", extra.get("duration"))
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None
            if duration is not None:
                requested_frames = int(round(duration * fps))
            else:
                requested_frames = None
        else:
            requested_frames = int(getattr(sampling, "num_frames", None) or 0)

        if requested_frames and video.ndim >= 2 and video.shape[1] > requested_frames:
            video = video[:, :requested_frames]
        output.output = (video, audio)

        # The uint8 quantize/permute/trim above is async on the worker; force
        # it to land before the engine can read the video buffer.
        current_omni_platform.synchronize()
        return output

    pipeline_cls.forward = forward_trim_sync
    logger.warning("MetaX: patched MiniMax-H3 forward trim + final GPU sync.")


def _wrap_encode_image_rank_local(vae_cls: type) -> None:
    """Force FL2VA keyframe VAE encode fully rank-local."""

    original_encode_image = vae_cls.encode_image

    def encode_image_rank_local(self, image):
        if int(getattr(self, "parallel_size", 1)) > 1:
            with self._rank_local_tiling():
                return original_encode_image(self, image)
        return original_encode_image(self, image)

    vae_cls.encode_image = encode_image_rank_local
    logger.warning("MetaX: patched MiniMax-H3 keyframe VAE encode to be rank-local.")


def _wrap_encode_video_rank_local(vae_cls: type) -> None:
    """Force Ref2VA reference-video VAE encode fully rank-local."""

    original_encode_video = vae_cls.encode_video

    def encode_video_rank_local(self, frames):
        if int(getattr(self, "parallel_size", 1)) > 1:
            with self._rank_local_tiling():
                return original_encode_video(self, frames)
        return original_encode_video(self, frames)

    vae_cls.encode_video = encode_video_rank_local
    logger.warning("MetaX: patched MiniMax-H3 reference-video VAE encode to be rank-local.")


def apply_h3_sync_patch() -> None:
    global _PATCHED, _RETRY_COUNT
    if _PATCHED:
        return

    try:
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            MiniMaxH3Pipeline,
        )
        from vllm_omni.diffusion.models.minimax_h3.vae import (
            MiniMaxH3VideoVAE,
        )
    except Exception:
        if _RETRY_COUNT < _MAX_RETRIES:
            _RETRY_COUNT += 1
            threading.Timer(1.0, apply_h3_sync_patch).start()
        else:
            logger.error(
                "h3 sync patch: vllm_omni never became ready after %d retries.",
                _MAX_RETRIES,
            )
        return

    _wrap_forward_trim_and_sync(MiniMaxH3Pipeline)
    _wrap_encode_image_rank_local(MiniMaxH3VideoVAE)
    _wrap_encode_video_rank_local(MiniMaxH3VideoVAE)
    _PATCHED = True
    logger.info("MiniMax-H3 MetaX minimal patches applied.")