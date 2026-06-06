"""#25 (Phase 3/4): scan Sierra's mirror locator for a Dirichlet L-function.

The L-function generalisation (Sierra arXiv:1404.4252 eq. 13.6 / 14.9): switch the
mirror reflection coefficients ``mu(n)/sqrt(n) -> chi(n) mu(n)/sqrt(n)`` for a
Dirichlet character ``chi``, and the forward locator ``|M'_z(n)|`` -- the truncated
partial sum of ``1/L(s, chi)`` -- peaks at the zeros of ``L(s, chi)``. The character
depends only on its modulus; the zeros come out (forward, not inverse). Real
characters (zeta, principal, quadratic) use the fast real kernel; a genuinely
complex character uses the ``weighted_locator`` kernel and shows the
asymmetric-in-E zeros of a complex L-function.

Prints the GPU-vs-CPU agreement (house rule), the wall-time, and the
forward-located peaks against the independently-computed L-zeros
(:func:`dirichlet.lfunction_zeros`).

    uv run python scripts/run_dirichlet_mirror.py                      # Dirichlet beta (mod 4)
    uv run python scripts/run_dirichlet_mirror.py --modulus 5 --index 1  # complex L(s,chi)
    uv run python scripts/run_dirichlet_mirror.py --modulus 7 --index 3 --n 40000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu import dirichlet as dl


def _local_maxima(y: np.ndarray, height: float) -> np.ndarray:
    interior = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > height)
    return np.nonzero(interior)[0] + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--modulus", type=int, default=4, help="character modulus (prime or 4)"
    )
    ap.add_argument("--index", type=int, default=1, help="character index")
    ap.add_argument("--n", type=int, default=8000, help="mirror truncation (terms)")
    ap.add_argument(
        "--e-lo", type=float, default=None, help="E-grid low (default: auto)"
    )
    ap.add_argument("--e-max", type=float, default=25.0)
    ap.add_argument("--step", type=float, default=0.01, help="E-grid step")
    ap.add_argument("--cpu", action="store_true", help="force the CPU locator")
    args = ap.parse_args()

    chi = dl.dirichlet_character(args.modulus, args.index)
    real = dl.is_real_character(chi)
    # A complex character's zeros are asymmetric in E -> scan negatives by default.
    e_lo = args.e_lo if args.e_lo is not None else (1.0 if real else -args.e_max)
    grid = np.arange(e_lo, args.e_max, args.step)
    w = dl.lfunction_weights(chi, args.n)

    kind = "real" if real else "complex"
    kernel = "mobius_locator" if real else "weighted_locator"
    print(
        f"Dirichlet-L mirror locator: chi mod {args.modulus} (index {args.index}, "
        f"{kind}), n={args.n} mirrors, {grid.size} E-points "
        f"in [{e_lo:g}, {args.e_max:g}]"
    )

    g_m = None
    t_gpu = None
    if not args.cpu:
        try:
            from zeta_spectral_gpu import dirac_mirror_gpu as gpu

            t0 = time.perf_counter()
            g_m = gpu.mobius_partial_sum_gpu(grid, args.n, weights=w)
            abs_m = np.abs(g_m)  # includes the device->host copy in the timing
            t_gpu = time.perf_counter() - t0
            print(f"  GPU scan: {t_gpu:.3f}s  ({kernel} kernel)")
        except ImportError:
            print("  (cupy unavailable -- CPU only)")

    t0 = time.perf_counter()
    c_m = dm.mobius_partial_sum(grid, args.n, weights=w)
    t_cpu = time.perf_counter() - t0
    if g_m is not None:
        agree = float(np.max(np.abs(g_m - c_m)))
        print(
            f"  CPU scan: {t_cpu:.3f}s   speedup {t_cpu / t_gpu:.1f}x   "
            f"GPU-vs-CPU max|dM|={agree:.2e}"
        )
    else:
        abs_m = np.abs(c_m)
        print(f"  CPU scan: {t_cpu:.3f}s")

    peaks = grid[_local_maxima(abs_m, 0.3 * np.log(args.n))]
    print("  computing independent L-zeros (mpmath) ...")
    true = dl.lfunction_zeros(chi, e_min=e_lo, e_max=args.e_max)

    print("\nforward-located peaks vs. independent L(s,chi) zeros (matched < 0.3):")
    print("     true E        located peak    |err|")
    matched = 0
    for g in true:
        j = int(np.argmin(np.abs(peaks - g))) if peaks.size else -1
        if j >= 0 and abs(peaks[j] - g) < 0.3:
            matched += 1
            print(f"   {g:11.5f}   {peaks[j]:11.5f}   {abs(peaks[j] - g):.4f}")
        else:
            print(f"   {g:11.5f}   {'(missed)':>11}")
    print(f"\nmatched {matched}/{true.size} zeros of L(s, chi mod {args.modulus})")


if __name__ == "__main__":
    main()
