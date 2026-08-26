# SPDX-License-Identifier: Apache-2.0
"""Pytest wrapper for the C600U MCDNN Conv3d tile bug repro."""

from __future__ import annotations

import pytest
import torch

from reproduce_c600u_mcdnn_conv3d import tile_vs_full_diff


def _is_c600u() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    return "MetaX" in name and "600" in name


requires_c600u = pytest.mark.skipif(
    not _is_c600u(),
    reason="MCDNN Conv3d tile bug is C600U-specific; requires a MetaX C600U GPU",
)


@requires_c600u
def test_cudnn_on_reproduces_conv3d_tile_bug():
    """cudnn ON must show a large tile-vs-full mismatch on C600U."""
    mean, _ = tile_vs_full_diff(True)
    assert mean != mean or mean > 1e-3, (
        "Expected the MCDNN Conv3d tile mismatch on C600U (mean diff > 1e-3), "
        "got %.8f. If the driver was fixed this test should be removed." % mean
    )


@requires_c600u
def test_cudnn_off_conv3d_tile_is_consistent():
    """Native torch conv (cudnn off) must be exactly consistent."""
    mean, _ = tile_vs_full_diff(False)
    assert mean < 1e-5, "Native conv should be exact, got mean diff %.8f" % mean
