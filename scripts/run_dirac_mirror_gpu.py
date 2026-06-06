"""Warm-up #25 (Phase 2): GPU scan of Sierra's prime-driven Moebius-mirror locator.

The GPU half of the forward locator (#25, Piece A). The CPU reference
``dirac_mirror.mobius_partial_sum`` materializes the full ``(len(E) x n)`` complex
phase matrix to scan ``|M'_z(n)|`` over an ``E``-grid; the hand-written kernel
``kernels/dirac_mirror.cu`` computes the same sum with one thread per ``E`` point
and an inner loop over ``k`` -- ``O(len(E) * n)`` work in ``O(len(E) + n)`` memory,
so the grid can be made fine and ``n`` large without the outer-product's memory
blow-up. fp64 throughout (the locator is an ``O(1)``, cancellation-free object --
the clean fp64 GPU win the #25 ruling anticipated, opposite the flagship).

Prints the GPU-vs-CPU agreement (house rule), the wall-time speedup, and the
forward-located zeros vs. the true ordinates (a downstream check; only ``mu(k)``
and ``k^{-z}`` go into the scan -- no zeros consumed).

    uv run python scripts/run_dirac_mirror_gpu.py
    uv run python scripts/run_dirac_mirror_gpu.py --n 40000 --e-max 80 --step 0.005
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu import dirac_mirror_gpu as gpu
from zeta_spectral_gpu.zeros import load_ordinates


def _local_maxima(y: np.ndarray, height: float) -> np.ndarray:
    """Indices of strict local maxima of ``y`` above ``height`` (no scipy needed)."""
    interior = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > height)
    return np.nonzero(interior)[0] + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=8000, help="mirror truncation (terms)")
    ap.add_argument("--e-lo", type=float, default=10.0)
    ap.add_argument("--e-max", type=float, default=50.0)
    ap.add_argument("--step", type=float, default=0.01, help="E-grid step")
    ap.add_argument(
        "--no-cpu-check",
        action="store_true",
        help="skip the numpy cross-check (the outer product can exhaust host RAM)",
    )
    args = ap.parse_args()

    grid = np.arange(args.e_lo, args.e_max, args.step)
    print(
        f"Sierra Moebius-mirror GPU locator  (n={args.n} mirrors, "
        f"{grid.size} E-points in [{args.e_lo:g}, {args.e_max:g}])"
    )

    t0 = time.perf_counter()
    gM = gpu.mobius_partial_sum_gpu(grid, args.n)
    absM = np.abs(gM)  # includes the device->host copy in the timing
    t_gpu = time.perf_counter() - t0
    print(f"  GPU scan: {t_gpu:.3f}s")

    if not args.no_cpu_check:
        t0 = time.perf_counter()
        cM = dm.mobius_partial_sum(grid, args.n)
        t_cpu = time.perf_counter() - t0
        agree = float(np.max(np.abs(gM - cM)))
        print(
            f"  CPU scan: {t_cpu:.3f}s   speedup {t_cpu / t_gpu:.1f}x   "
            f"GPU-vs-CPU max|dM|={agree:.2e}"
        )

    peaks = grid[_local_maxima(absM, 0.5 * np.log(args.n))]
    true = load_ordinates(80, source="mpmath")
    true = true[(true >= args.e_lo) & (true < args.e_max)]
    matched = []
    for g in true:
        if peaks.size:
            j = int(np.argmin(np.abs(peaks - g)))
            if abs(peaks[j] - g) < 0.3:
                matched.append((g, float(peaks[j])))

    print(
        f"\nforward-located zeros vs. true ordinates "
        f"(matched {len(matched)}/{true.size} in [{args.e_lo:g}, {args.e_max:g}]):"
    )
    print("     true gamma   located peak    |err|")
    for g, p in matched:
        print(f"   {g:11.5f}   {p:11.5f}   {abs(p - g):.4f}")


if __name__ == "__main__":
    main()
