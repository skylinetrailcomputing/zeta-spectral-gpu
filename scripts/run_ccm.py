"""Flagship #8: CPU multiprecision reference for the Connes–Consani–Moscovici
finite-cutoff operator — reproduce the source's convergence-to-the-zeros table.

Assembles the Weil quadratic form ``QW_lambda^N`` in ``mpmath`` at ~200 digits,
extracts the rank-one perturbed scaling operator from its minimal eigenvector,
and compares that operator's spectrum to the true ordinates ``t_k`` of
``zeta(1/2 + i t)``. The errors ``|eig_k - t_k|`` should match the §6 table of
arXiv:2511.22755 (e.g. for ``lambda = sqrt(13)``: ``~2.4e-55`` at ``k = 1``,
``~2.0e-3`` at ``k = 50``), and shrink fast as the prime cutoff ``x = lambda^2``
grows.

Forward, not inverse: the operator is built from the primes (a von Mangoldt sum
cut at ``k <= lambda^2``); the zeros are consumed only to *check* the output. This
is CPU/mpmath by necessity — the agreements are far below fp64 epsilon (see
CLAUDE.md and knowledge/ccm-operator.md).

    uv run python scripts/run_ccm.py                       # headline: x in {12,13,14}
    uv run python scripts/run_ccm.py --x 13 --count 50     # one cutoff
    uv run python scripts/run_ccm.py --x 9 --count 20 --N 120   # the Fig. 1 case
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import ccm, plots

DATA = Path(__file__).resolve().parent.parent / "data"

# Source §6 upper bounds on |eig_1 - t_1| and |eig_50 - t_50| at N=120, for a
# sanity print only (we never fit to these). x -> (k=1 bound, k=50 bound).
SOURCE_BOUNDS = {
    12: (3.41e-50, 9.02e-2),
    13: (2.44e-55, 2.04e-3),
    14: (1.07e-60, 4.78e-6),
}


def _lambda_label(x: int) -> str:
    return rf"$x={x}$ ($\lambda=\sqrt{{{x}}}$)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--x",
        type=int,
        nargs="+",
        default=[12, 13, 14],
        help="prime cutoffs x = lambda^2 (lambda = sqrt(x))",
    )
    ap.add_argument("--N", type=int, default=120, help="truncation; dim = 2N+1")
    ap.add_argument("--count", type=int, default=50, help="zeros to match")
    ap.add_argument("--dps", type=int, default=210, help="mpmath precision")
    ap.add_argument("--out", type=Path, default=DATA / "ccm_convergence.png")
    args = ap.parse_args()

    print(
        f"CCM finite-cutoff convergence  (N={args.N}, dim={2 * args.N + 1}, "
        f"dps={args.dps}, matching first {args.count} zeros)"
    )
    errors_by_lambda: dict[str, np.ndarray] = {}
    for x in args.x:
        t0 = time.perf_counter()
        res = ccm.converge(args.N, mp.sqrt(x), args.count, dps=args.dps)
        dt = time.perf_counter() - t0
        errors_by_lambda[_lambda_label(x)] = ccm.errors_as_float(res.errors)

        e1 = res.errors[0]
        elast = res.errors[-1]
        print(f"\nx = {x}  (lambda = sqrt({x}))   [{dt:.0f}s]")
        print(
            f"  eps_N = {mp.nstr(res.eigenvalue, 4)}   "
            f"parity residual = {mp.nstr(res.parity_residual, 3)}"
        )
        print(
            f"  |eig_1  - t_1 |  = {mp.nstr(e1, 4)}"
            + (
                f"   (source <= {SOURCE_BOUNDS[x][0]:.2e})"
                if x in SOURCE_BOUNDS
                else ""
            )
        )
        print(
            f"  |eig_{args.count} - t_{args.count}|  = {mp.nstr(elast, 4)}"
            + (
                f"   (source <= {SOURCE_BOUNDS[x][1]:.2e})"
                if x in SOURCE_BOUNDS and args.count == 50
                else ""
            )
        )
        # A few intermediate rows so the growth-with-k is visible.
        idxs = sorted({0, 9, 19, 29, 39, args.count - 1})
        print("    k     |eig_k - t_k|")
        for i in idxs:
            if i < len(res.errors):
                print(f"   {i + 1:3d}     {mp.nstr(res.errors[i], 6)}")

    out = plots.ccm_convergence_figure(
        errors_by_lambda,
        N=args.N,
        out_path=args.out,
        title=f"CCM finite-cutoff convergence to the zeros ($N={args.N}$)",
    )
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
