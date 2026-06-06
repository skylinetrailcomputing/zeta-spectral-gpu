"""Dirichlet-``L`` mirror locator (issue #25/#42) — packaged visualizer (#60).

Scan Sierra's prime-driven mirror locator for a Dirichlet ``L``-function and,
optionally, plot it. The ``L``-function generalisation (arXiv:1404.4252
eq. 13.6 / 14.9): switch the mirror reflection coefficients
``mu(n)/sqrt(n) -> chi(n) mu(n)/sqrt(n)`` for a Dirichlet character ``chi``, and the
forward locator ``|M'_z(n)|`` — the truncated partial sum of ``1/L(s, chi)`` — peaks
at the zeros of ``L(s, chi)``. The character depends only on its modulus; the zeros
come out (forward, not inverse). Real characters (zeta, principal, quadratic) use the
fast real kernel; a genuinely complex character uses the ``weighted_locator`` kernel
and shows the asymmetric-in-E zeros of a complex L-function.

The scan -> peaks -> score core now lives in :mod:`zeta_spectral_gpu.dirichlet_locator`
(shared with the notebook / any web demo); this CLI wires it to argparse, prints the
GPU-vs-CPU agreement (house rule), the wall-time, and the forward-located peaks
against the independently-computed L-zeros, and optionally saves the figure.

    uv run python scripts/run_dirichlet_mirror.py                       # Dirichlet beta (mod 4)
    uv run python scripts/run_dirichlet_mirror.py --modulus 1 --plot    # Riemann zeta, with a PNG
    uv run python scripts/run_dirichlet_mirror.py --modulus 5 --index 1 # complex L(s,chi)
    uv run python scripts/run_dirichlet_mirror.py --modulus 7 --index 3 --n 40000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import dirichlet as dl
from zeta_spectral_gpu import dirichlet_locator as dlc

DATA = Path(__file__).resolve().parent.parent / "data"


def _timed_scan(chi, n, grid, *, prefer_gpu):
    t0 = time.perf_counter()
    vals, backend = dlc.scan_locator(chi, n, grid, prefer_gpu=prefer_gpu)
    return vals, backend, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--modulus", type=int, default=4, help="character modulus (prime, 4, or 1=zeta)"
    )
    ap.add_argument("--index", type=int, default=1, help="character index")
    ap.add_argument("--n", type=int, default=8000, help="mirror truncation (terms)")
    ap.add_argument(
        "--e-lo", type=float, default=None, help="E-grid low (default: auto)"
    )
    ap.add_argument("--e-max", type=float, default=25.0)
    ap.add_argument("--step", type=float, default=0.01, help="E-grid step")
    ap.add_argument("--cpu", action="store_true", help="force the CPU locator")
    ap.add_argument("--plot", action="store_true", help="save a |M'_z(E)| figure")
    ap.add_argument(
        "--out", type=Path, default=None, help="figure path (default: data/...)"
    )
    args = ap.parse_args()

    n = args.n
    if args.modulus == 1:
        chi = np.ones(1, dtype=np.complex128)  # trivial period-1 character -> zeta
        name = "zeta"
    else:
        chi = dl.dirichlet_character(args.modulus, args.index)
        name = f"chi mod {args.modulus} (index {args.index})"
    real = dl.is_real_character(chi)
    # A complex character's zeros are asymmetric in E -> scan negatives by default.
    e_lo = args.e_lo if args.e_lo is not None else (1.0 if real else -args.e_max)
    grid = np.arange(e_lo, args.e_max, args.step)

    kind = "real" if real else "complex"
    kernel = "mobius_locator" if real else "weighted_locator"
    print(
        f"Dirichlet-L mirror locator: {name}, {kind}, n={n} mirrors, "
        f"{grid.size} E-points in [{e_lo:g}, {args.e_max:g}]"
    )

    # --- the scan(s): GPU first (timed), CPU for the house GPU-vs-CPU agreement ---
    gpu_vals, t_gpu, cpu_vals, t_cpu = None, None, None, None
    if not args.cpu:
        vals, backend, dt = _timed_scan(chi, n, grid, prefer_gpu=True)
        if backend == "gpu":
            gpu_vals, t_gpu = vals, dt
            print(f"  GPU scan: {t_gpu:.3f}s  ({kernel} kernel)")
        else:  # cupy unavailable -> the call already produced a CPU scan; reuse it
            cpu_vals, t_cpu = vals, dt
            print("  (cupy unavailable -- CPU only)")

    if gpu_vals is not None:
        cpu_vals, _, t_cpu = _timed_scan(chi, n, grid, prefer_gpu=False)
        agree = float(np.max(np.abs(gpu_vals - cpu_vals)))
        print(
            f"  CPU scan: {t_cpu:.3f}s   speedup {t_cpu / t_gpu:.1f}x   "
            f"GPU-vs-CPU max|dM|={agree:.2e}"
        )
        values, backend_used = gpu_vals, "gpu"
    else:
        if cpu_vals is None:  # --cpu path: no GPU attempt was made
            cpu_vals, _, t_cpu = _timed_scan(chi, n, grid, prefer_gpu=False)
            print(f"  CPU scan: {t_cpu:.3f}s")
        values, backend_used = cpu_vals, "cpu"

    # --- score against the INDEPENDENT L-zeros (mpmath); never fed into the scan --
    print("  computing independent L-zeros (mpmath) ...")
    true = dl.lfunction_zeros(chi, e_min=e_lo, e_max=args.e_max)
    res = dlc.score_scan(
        grid,
        np.abs(values),
        height=dlc.peak_threshold(n),
        true_zeros=true,
        char=chi,
        backend=backend_used,
    )

    print("\nforward-located peaks vs. independent L(s,chi) zeros (matched < 0.3):")
    print("     true E        located peak    |err|")
    for m in res.matches:
        if m.matched:
            print(f"   {m.true_E:11.5f}   {m.peak_E:11.5f}   {m.error:.4f}")
        else:
            print(f"   {m.true_E:11.5f}   {'(missed)':>11}")
    print(f"\nmatched {res.matched_count}/{res.true_zeros.size} zeros of L(s, {name})")

    if args.plot:
        from zeta_spectral_gpu import plots

        out = (
            args.out
            or DATA / f"dirichlet_locator_mod{args.modulus}_idx{args.index}.png"
        )
        out = plots.dirichlet_locator_figure(
            res.grid,
            res.abs_m,
            res.peaks,
            res.true_zeros,
            out_path=out,
            n=n,
            modulus=args.modulus,
            index=args.index,
            height=res.height,
        )
        print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
