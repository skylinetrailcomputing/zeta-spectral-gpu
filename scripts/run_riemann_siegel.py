"""Warm-up #55: GPU Riemann-Siegel evaluator for the Hardy Z function on the line.

Evaluates ``Z(t) = e^{i theta(t)} zeta(1/2 + i t)`` on a dense height grid with the
hand-written ``kernels/riemann_siegel.cu`` main-sum kernel (one thread per height,
``O(sqrt(t))`` terms each), then reads the on-line zeta zeros off the sign changes
of ``Z`` -- a forward use (the zeros are an output, only scored against mpmath).

Prints the GPU-vs-CPU agreement (house rule), the wall-time speedup, the accuracy
against ``mpmath.siegelz``, and the forward-located zeros vs. mpmath's residual.

    uv run python scripts/run_riemann_siegel.py
    uv run python scripts/run_riemann_siegel.py --t-lo 1000 --t-hi 5000 --step 0.002
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from zeta_spectral_gpu import riemann_siegel as rs


def _sign_change_zeros(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear-interpolated sign-change locations of ``z(t)`` -- the forward locator."""
    s = np.signbit(z)
    idx = np.nonzero(s[:-1] != s[1:])[0]
    z0, z1 = z[idx], z[idx + 1]
    t0, t1 = t[idx], t[idx + 1]
    return t0 - z0 * (t1 - t0) / (z1 - z0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--t-lo", type=float, default=1.0e6)
    ap.add_argument("--t-hi", type=float, default=1.0e6 + 2000.0)
    ap.add_argument("--step", type=float, default=0.005, help="height-grid step")
    ap.add_argument("--correction-terms", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument(
        "--no-cpu-check", action="store_true", help="skip the numpy cross-check"
    )
    args = ap.parse_args()

    grid = np.arange(args.t_lo, args.t_hi, args.step)
    n_terms = int(np.floor(np.sqrt(grid.max() / rs.TWO_PI)))
    print(
        f"Riemann-Siegel GPU Z-evaluator  ({grid.size} heights in "
        f"[{args.t_lo:g}, {args.t_hi:g}], ~{n_terms} main-sum terms each, "
        f"{args.correction_terms} correction terms)"
    )

    try:
        from zeta_spectral_gpu import riemann_siegel_gpu as gpu

        # The kernel owns the O(sqrt(t)) main sum -- time it against the numpy
        # reference (the asymptotic remainder is shared O(n) host work either way).
        gpu.main_sum_gpu(grid[:8])  # warm up the JIT
        t0 = time.perf_counter()
        ms_g = gpu.main_sum_gpu(grid)
        t_gpu = time.perf_counter() - t0
        zg = ms_g + np.asarray(
            rs.rs_remainder(grid, correction_terms=args.correction_terms)
        )
        have_gpu = True
    except Exception as exc:  # noqa: BLE001 - CPU-only box: fall back gracefully
        print(f"  (GPU unavailable: {exc}; using CPU)")
        zg = rs.hardy_z(grid, correction_terms=args.correction_terms)
        t_gpu, have_gpu = float("nan"), False

    if have_gpu and not args.no_cpu_check:
        t0 = time.perf_counter()
        ms_c = rs.main_sum(grid)
        t_cpu = time.perf_counter() - t0
        agree = float(np.max(np.abs(ms_g - ms_c)))
        print(
            f"  main sum  GPU {t_gpu:.3f}s  CPU {t_cpu:.3f}s  "
            f"speedup {t_cpu / t_gpu:.1f}x   GPU-vs-CPU max|d|={agree:.2e}"
        )

    # Accuracy against the mpmath ground truth on a sample of heights.
    import mpmath as mp

    sample = grid[:: max(1, grid.size // 8)]
    acc = max(
        abs(
            float(rs.hardy_z(t, correction_terms=args.correction_terms))
            - float(mp.siegelz(t))
        )
        for t in sample
    )
    print(f"  vs mpmath.siegelz (sampled): max|dZ|={acc:.2e}")

    located = _sign_change_zeros(grid, np.asarray(zg))
    print(
        f"\nforward-located on-line zeros: {located.size} sign changes of Z in "
        f"[{args.t_lo:g}, {args.t_hi:g}]"
    )
    print("   located gamma   |Z_mpmath(gamma)|   (should be ~0)")
    for g in located[:8]:
        print(f"   {g:13.6f}   {abs(float(mp.siegelz(g))):.2e}")


if __name__ == "__main__":
    main()
