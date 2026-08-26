#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the MetaX C600U MCDNN Conv3d tile bug.

Bug: on C600U (sdk3.8.2.8 / torch 2.8, MCDNN), a bf16 ``nn.Conv3d`` with a
large output channel count produces different results for a small tile input
vs. the corresponding region of a larger input:

    Conv3d(16, 384, (3, 3, 3), padding=(1, 0, 0))  # bf16, cudnn ON
    tile  (1, 16, 1, 32, 32)  vs  full  (1, 16, 1, 60, 104)
    -> interior mean abs diff ~0.12-0.15  (should be 0)

With ``torch.backends.cudnn.enabled = False`` (native torch conv) the same
comparison is exactly 0.  The bug shows up in vllm-omni Wan2.2 VAE tiled
decoding as edge color blur / chromatic aberration on C600U.

Run directly:
    python reproduce_c600u_mcdnn_conv3d.py

Or as a pytest:
    pytest tests/unit/test_c600u_mcdnn_conv3d.py -v
"""

from __future__ import annotations

import platform
import subprocess
import sys

import torch


def env_info() -> str:
    lines = [
        "python: %s" % platform.python_version(),
        "torch : %s" % torch.__version__,
    ]
    if torch.cuda.is_available():
        lines.append("cuda devices: %d" % torch.cuda.device_count())
        lines.append("device 0    : %s" % torch.cuda.get_device_name(0))
    try:
        import vllm_omni_metax  # noqa: F401

        lines.append("vllm-omni-metax: %s" % getattr(vllm_omni_metax, "__version__", "n/a"))
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["/opt/mxdriver/bin/mx-smi", "--show-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            lines.append("mx-smi:\n%s" % "\n".join(out.stdout.splitlines()[:6]))
    except Exception:
        pass
    return "\n".join(lines)


def tile_vs_full_diff(
    cudnn_enabled: bool,
    *,
    cin: int = 16,
    cout: int = 384,
    kernel: tuple[int, int, int] = (3, 3, 3),
    th: int = 32,
    tw: int = 32,
    fh: int = 60,
    fw: int = 104,
) -> tuple[float, float]:
    """Run the bf16 Conv3d on a tile slice vs the full input.

    Returns (interior mean abs diff, interior max abs diff) between
    ``conv(tile)`` and the corresponding region of ``conv(full)``.
    For a correct conv kernel this must be 0.
    """
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.manual_seed(0)

    conv = torch.nn.Conv3d(cin, cout, kernel, padding=(1, 0, 0)).to("cuda", torch.bfloat16).eval()
    full = torch.randn(1, cin, 1, fh, fw, device="cuda", dtype=torch.bfloat16)
    tile = full[..., :th, :tw]

    with torch.no_grad():
        out_tile = conv(tile)
        out_full = conv(full)

    oh, ow = out_tile.shape[-2], out_tile.shape[-1]
    diff = (out_tile.float() - out_full.float()[..., :oh, :ow]).abs()
    # Skip the 1px spatial border (padding boundary is expected to differ).
    interior = diff[:, :, :, 1 : oh - 1, 1 : ow - 1]
    return interior.mean().item(), interior.max().item()


def main() -> int:
    print("=== MetaX C600U MCDNN Conv3d tile bug repro ===")
    print(env_info())

    if not torch.cuda.is_available() or "MetaX" not in torch.cuda.get_device_name(0):
        print("SKIP: not a MetaX GPU (need C600U to reproduce).")
        return 0

    on_mean, on_max = tile_vs_full_diff(True)
    off_mean, off_max = tile_vs_full_diff(False)

    print("")
    print("Conv3d(16, 384, (3,3,3), padding=(1,0,0)) bf16")
    print("  tile (1,16,1,32,32) vs full (1,16,1,60,104), interior region:")
    print("    cudnn ON : mean diff = %.8f  max diff = %.4f" % (on_mean, on_max))
    print("    cudnn OFF: mean diff = %.8f  max diff = %.4f" % (off_mean, off_max))

    on_broken = on_mean != on_mean or on_mean > 1e-3  # NaN or large diff
    if on_broken and off_mean < 1e-5:
        print("\nRESULT: BUG REPRODUCED (cudnn ON mismatches/NaN, native conv is correct)")
        return 1
    if not on_broken:
        print("\nRESULT: NOT reproduced (cudnn ON matches) - bug may be fixed or config differs")
        return 0
    print("\nRESULT: ambiguous - both paths differ; inspect output above")
    return 2


if __name__ == "__main__":
    sys.exit(main())
