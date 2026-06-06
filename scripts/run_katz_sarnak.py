"""Katz-Sarnak family statistics for the quadratic Dirichlet L-functions (issue #51).

Take the family of quadratic Dirichlet ``L``-functions ``L(s, chi_d)`` over
fundamental discriminants ``d`` (the character ``chi_d(n) = (d | n)`` is a Kronecker
symbol), locate the low-lying zeros of each, rescale by the conductor, and pool them
into the family **1-level density** ``W(x)``. Katz-Sarnak universality predicts the
family is **symplectic** -- ``W(x) -> 1 - sin(2 pi x)/(2 pi x)``, suppressed at the
central point -- distinct from unitary (flat) and even-orthogonal (enhanced). The
zeros are produced independently per member (mpmath) and only compared; nothing is
fit (forward, not inverse).

The per-member zero scan is embarrassingly parallel over the family -- the GPU
angle. ``--locate`` additionally runs the prime-driven mirror locator
(:mod:`dirichlet_locator`, GPU with CPU fallback) on a sample member and reports
the GPU-vs-CPU agreement; the precise statistic still uses the mpmath ground truth.

    uv run python scripts/run_katz_sarnak.py
    uv run python scripts/run_katz_sarnak.py --d-max 80 --plot
    uv run python scripts/run_katz_sarnak.py --d-max 60 --locate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import katz_sarnak as ks

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-max", type=int, default=60, help="max |d| (family bound)")
    ap.add_argument("--x-max", type=float, default=3.0, help="rescaled-height cutoff")
    ap.add_argument("--n-bins", type=int, default=12, help="1-level density bins")
    ap.add_argument("--x-disc", type=float, default=1.0, help="discrimination window")
    ap.add_argument("--dps", type=int, default=15, help="mpmath precision")
    ap.add_argument("--locate", action="store_true", help="also run the GPU locator")
    ap.add_argument("--plot", action="store_true", help="save the density figure")
    ap.add_argument("--out", type=Path, default=None, help="figure path")
    args = ap.parse_args()

    family = ks.fundamental_discriminants(args.d_max)
    n_real = sum(d > 0 for d in family)
    print(
        f"quadratic Dirichlet family: {len(family)} members, |d| <= {args.d_max} "
        f"({n_real} real d>0, {len(family) - n_real} imaginary d<0)"
    )

    print("  computing independent low-lying zeros (mpmath) ...")
    t0 = time.perf_counter()
    centres, w, n_members = ks.family_one_level_density(
        family, x_max=args.x_max, n_bins=args.n_bins, dps=args.dps
    )
    dt = time.perf_counter() - t0
    print(f"  pooled the family in {dt:.1f}s")

    print("\n  x      W_emp   W_Sp    W_U    W_SO+")
    sp = ks.symplectic_density(centres)
    u = ks.unitary_density(centres)
    so = ks.orthogonal_even_density(centres)
    for xc, we, ws, wu, wso in zip(centres, w, sp, u, so):
        print(f"  {xc:4.2f}   {we:5.2f}   {ws:5.2f}   {wu:5.2f}   {wso:5.2f}")

    bulk = w[centres > 1.5]
    distances = ks.symmetry_distances(centres, w, x_disc=args.x_disc)
    verdict = min(distances, key=distances.get)
    print(f"\n  bulk mean W (x>1.5): {bulk.mean():.3f}  (kernels -> 1)")
    print(f"  RMS distance to kernels on x <= {args.x_disc:g}:")
    for name, d in sorted(distances.items(), key=lambda kv: kv[1]):
        mark = "  <-- closest" if name == verdict else ""
        print(f"    {name:16s} {d:.4f}{mark}")
    print(f"\n  forward verdict: the quadratic family is {verdict.upper()}")

    if args.locate:
        d = next(x for x in family if x < 0)  # a sample member
        grid = np.arange(0.5, 18.0, 0.01)
        t0 = time.perf_counter()
        peaks, backend = ks.locate_member_zeros(d, 6000, grid, prefer_gpu=True)
        print(
            f"\n  locator on d={d} (chi mod {abs(d)}): {backend} backend, "
            f"{peaks.size} peaks in {time.perf_counter() - t0:.3f}s"
        )
        if backend == "gpu":
            cpu_peaks, _ = ks.locate_member_zeros(d, 6000, grid, prefer_gpu=False)
            agree = (
                float(np.max(np.abs(np.sort(peaks) - np.sort(cpu_peaks))))
                if peaks.size == cpu_peaks.size
                else float("nan")
            )
            print(f"  GPU-vs-CPU located-peak max|dE|={agree:.2e}")

    if args.plot:
        from zeta_spectral_gpu import plots

        out = args.out or DATA / f"katz_sarnak_density_dmax{args.d_max}.png"
        out = plots.katz_sarnak_density_figure(
            centres, w, out_path=out, n_members=n_members, x_disc=args.x_disc
        )
        print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
