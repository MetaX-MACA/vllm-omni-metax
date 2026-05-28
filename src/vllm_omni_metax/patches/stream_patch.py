# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@contextmanager
def use_current_stream_for_runner_init():
    """
    Avoid new CUDA stream creation during vLLM runner init on MetaX.

    MACA can hang on torch.cuda.Stream() in vLLM GPUModelRunner.__init__.
    Reuse the current stream only while the runner is being constructed.
    """
    original_stream = torch.cuda.Stream
    logged = False

    def current_stream_factory(*args, **kwargs):
        nonlocal logged
        if not logged:
            logger.warning(
                "MetaX: using current CUDA stream during GPUModelRunner init."
            )
            logged = True

        # current_stream() may internally wrap active stream through torch.cuda.Stream,
        # so restore the original constructor just for this call.
        torch.cuda.Stream = original_stream
        try:
            return torch.cuda.current_stream()
        finally:
            torch.cuda.Stream = current_stream_factory

    torch.cuda.Stream = current_stream_factory
    try:
        yield
    finally:
        torch.cuda.Stream = original_stream
