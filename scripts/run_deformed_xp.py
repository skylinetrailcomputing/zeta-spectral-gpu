"""Warm-up #23: Sierra's deformed-``xp`` operator — a forward candidate spectrum.

Computes the eigenenergies of ``H = x(p + l_p^2/p)`` (Sierra & Rodríguez-Laguna,
PRL 106, 200201, 2011; arXiv:1102.5356) as the real roots of the Bessel-K secular
equation (eq. 14) at the Riemann self-adjoint extension ``theta = pi/4``, then
checks the forward sanity condition: the spectrum's staircase ``N(E)`` reproduces
the smooth Riemann–von Mangoldt count, i.e. the operator reproduces the *average*
Riemann zeros.

Forward, not inverse (see README / project-framing): the operator is a geometric
deformation of ``xp`` — no primes, no zeros consumed. The zeros appear only as an
*output* comparison here; the verdict that the spectrum reproduces the mean
density but NOT the GUE fluctuations is the warm-up's point, and the fluctuation
statistics themselves are the companion issue (#24).

    uv run python scripts/run_deformed_xp.py
    uv run python scripts/run_deformed_xp.py --n 50 --theta 0.7853981633974483
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import plots, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=30, help="eigenvalues to compute")
    ap.add_argument(
        "--theta",
        type=float,
        default=dxp.THETA_RIEMANN,
        help="self-adjoint extension angle (default pi/4, the Riemann case)",
    )
    ap.add_argument("--dps", type=int, default=None, help="mpmath precision override")
    ap.add_argument("--scan-step", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=DATA / "deformed_xp.png")
    ap.add_argument("--refresh", action="store_true", help="ignore cached spectrum")
    ap.add_argument(
        "--compare-ordinates",
        type=int,
        default=8,
        help="rows comparing E_n to the actual zeta ordinates (output-only)",
    )
    args = ap.parse_args()

    print(
        f"Sierra deformed-xp forward spectrum  "
        f"(n={args.n}, theta={args.theta / np.pi:.4g}*pi, "
        f"dps={args.dps or dxp.required_dps(args.n)})"
    )
    t0 = time.perf_counter()
    spec = dxp.secular_spectrum(
        args.n,
        theta=args.theta,
        dps=args.dps,
        scan_step=args.scan_step,
        refresh=args.refresh,
    )
    avg = dxp.average_zeros(args.n)
    print(
        f"  {spec.size} eigenvalues in [{spec[0]:.3f}, {spec[-1]:.3f}]  "
        f"[{time.perf_counter() - t0:.0f}s]"
    )

    # Forward sanity: the analytic-unfolded spacings have unit mean (the staircase
    # tracks the smooth term), and the staircase offset N_bar(E_n) - (n - 1/2) is
    # small and shrinking -> the spectrum reproduces the average zero density.
    w = dxp.analytic_unfold(spec)
    s = np.diff(w)
    print(
        f"\nforward sanity: mean unfolded spacing = {s.mean():.4f} "
        f"(GUE/Poisson alike -> 1; here density matches by construction)"
    )
    print(
        f"                staircase offset N_bar(E_n)-(n-1/2): "
        f"first={float(zeros.smooth_count(spec[0])) - 0.5:+.3f}  "
        f"last={float(zeros.smooth_count(spec[-1])) - (spec.size - 0.5):+.3f}"
    )

    # Output-only comparison: the levels track the AVERAGE zeros, not the actual
    # (fluctuating) ordinates -- that gap is precisely the missing GUE physics.
    k = min(args.compare_ordinates, spec.size)
    gammas = np.array([float(mp.zetazero(j).imag) for j in range(1, k + 1)])
    print("\n   n    E_n       avg-zero    E_n-avg    actual g_n   E_n-g_n")
    for i in range(k):
        print(
            f"  {i + 1:2d}  {spec[i]:8.4f}  {avg[i]:8.4f}  {spec[i] - avg[i]:+8.4f}  "
            f"  {gammas[i]:8.4f}  {spec[i] - gammas[i]:+8.4f}"
        )
    print(
        "\nverdict: spectrum reproduces the AVERAGE zeros (gap to avg shrinks with "
        "n);\n         the residual scatter vs the actual ordinates is the GUE "
        "fluctuation\n         this model does NOT carry (the #24 readout)."
    )

    out = plots.deformed_xp_staircase_figure(
        spec,
        out_path=args.out,
        average_zeros=avg,
        title=(
            r"Sierra deformed-$xp$ ($\theta=\pi/4$): $N(E)$ reproduces the "
            rf"average zeros (n={spec.size})"
        ),
    )
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
