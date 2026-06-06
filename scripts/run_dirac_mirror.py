"""Warm-up #25: Sierra's prime-driven massless-Dirac model — a *forward* locator.

The Riemann zeros emerge from a purely number-theoretic object: the truncated
Möbius–Dirichlet sum (Sierra, arXiv:1404.4252, eq. 12.20; reviewed in 1601.01797)

    M'_z(n) = sum_{k<=n} mu(k) k^{-(1/2 + iE)},

whose magnitude *grows with the truncation n* at a Riemann zero and stays bounded
elsewhere (Fig. 14). The mirrors sit at sqrt(n) with reflection mu(n)/sqrt(n), so
a prime p contributes a "periodic orbit" of period log p — Berry's picture made
concrete. This script:

  1. scans ``|M'_z(n)|`` over a real ``E``-grid and locates the zeros as its peaks
     (forward: only mu(k) and k^{-z} go in; zeros come out),
  2. compares the located peaks to the true ordinates (a downstream check),
  3. verifies the growth-rate prediction ``|M'_z(n)| ~ log n / |Z'(E)|``, and
  4. shows the bound-state bridge (Fig. 4): the normalizable amplitude decays at a
     zero (with the tuned boundary phase) but not in the continuum.

Forward, not inverse: the ``vartheta`` tuning in step 4 uses ``theta(E)`` (an
explicit function) only to validate the bound-state machinery — it is *not* used
to find zeros. The locator (steps 1–3) consumes no zeros at all. See
``_private/issue-25-forward-ruling.md``.

    uv run python scripts/run_dirac_mirror.py
    uv run python scripts/run_dirac_mirror.py --n 4000 --e-max 60 --step 0.02
"""

from __future__ import annotations

import argparse

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu.zeros import load_ordinates


def _local_maxima(y: np.ndarray, height: float) -> np.ndarray:
    """Indices of strict local maxima of ``y`` above ``height`` (no scipy needed)."""
    interior = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > height)
    return np.nonzero(interior)[0] + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=3000, help="mirror truncation (terms)")
    ap.add_argument("--e-lo", type=float, default=10.0)
    ap.add_argument("--e-max", type=float, default=50.0)
    ap.add_argument("--step", type=float, default=0.02, help="E-grid step")
    ap.add_argument("--eps", type=float, default=0.25, help="mirror transparency scale")
    args = ap.parse_args()

    mu = dm.mobius_sieve(args.n)
    grid = np.arange(args.e_lo, args.e_max, args.step)

    print(f"Sierra Möbius-mirror locator  (n={args.n} mirrors, |M'_z(n)| on a grid)")
    print(
        f"  scanning E in [{args.e_lo:g}, {args.e_max:g}] step {args.step:g} "
        f"({grid.size} points)"
    )

    absM = np.abs(dm.mobius_partial_sum(grid, args.n, mu=mu))
    # A zero shows up as a peak with |M'| ~ log n / |Z'| >> the O(1) background.
    threshold = 0.5 * np.log(args.n)  # ~half the predicted peak height at |Z'|~1
    peaks = grid[_local_maxima(absM, threshold)]

    true = load_ordinates(40, source="mpmath")
    true = true[true < args.e_max]
    matched = []
    for g in true:
        if peaks.size:
            j = int(np.argmin(np.abs(peaks - g)))
            if abs(peaks[j] - g) < 0.3:
                matched.append((g, float(peaks[j])))

    print(
        f"\nforward-located zeros vs. true ordinates "
        f"(matched {len(matched)}/{true.size} below E={args.e_max:g}):"
    )
    print("     true gamma   located peak    |err|")
    for g, p in matched:
        print(f"   {g:11.5f}   {p:11.5f}   {abs(p - g):.4f}")

    print("\ngrowth at a zero vs. between zeros  (|M'_z(n)|, Fig. 14):")
    tr = np.array([args.n // 4, args.n // 2, args.n])
    for label, E in [
        ("zero g1", float(true[0])),
        ("zero g2", float(true[1])),
        ("midpoint", float(0.5 * (true[0] + true[1]))),
    ]:
        prof = dm.growth_profile(E, tr, mu=mu)
        vals = "  ".join(f"{v:6.3f}" for v in prof)
        print(f"   {label:9s} E={E:8.4f}:  n={tr[0]}->{tr[-1]}:  {vals}")

    print("\ngrowth-rate check  |M'_z(n)| ~ log n / |Z'(E)|  (eq. 12.30):")
    for k in range(min(3, true.size)):
        E = float(true[k])
        predicted = np.log(args.n) / abs(dm.hardy_z_prime(E))
        measured = float(dm.growth_profile(E, np.array([args.n]), mu=mu)[0])
        print(
            f"   gamma={E:8.4f}:  predicted {predicted:6.3f}   "
            f"measured {measured:6.3f}   ratio {measured / predicted:5.2f}"
        )

    print("\nbound-state bridge (Fig. 4): normalizable amplitude tail mean")
    tr2 = np.arange(10, args.n + 1)
    amp_zero = dm.normalizable_amplitude(float(true[0]), tr2, eps=args.eps, mu=mu)
    amp_cont = dm.normalizable_amplitude(
        float(0.5 * (true[0] + true[1])), tr2, eps=args.eps, vartheta=np.pi, mu=mu
    )
    print(
        f"   at zero g1 (tuned vartheta):  {amp_zero[-200:].mean():.4f}  (decays -> 0)"
    )
    print(
        f"   in continuum (vartheta=pi):   {amp_cont[-200:].mean():.4f}  (stays O(1))"
    )


if __name__ == "__main__":
    main()
