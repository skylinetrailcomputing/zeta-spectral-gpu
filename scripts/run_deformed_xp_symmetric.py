"""Warm-up #59: Berry-Keating x<->p-symmetric deformed-``xp`` model.

Computes the *semiclassical* (Bohr-Sommerfeld) counting function of
``H_II = (x + l_x^2/x)(p + l_p^2/p)`` -- the x<->p-symmetric sibling of Sierra's
``H_I = x(p + l_p^2/p)`` -- and compares it to ``H_I``'s count (eq. 5.17) and to the
average (smooth Riemann-von Mangoldt) zero count.

Why semiclassical and not a secular spectrum: ``H_II``'s metric is curved
(unlike ``H_I``'s flat Rindler metric), so it has *no* closed-form Bessel-K secular
equation -- the #23 reference-first template does not carry over. The area-based
count is what is exactly computable. See
``zeta_spectral_gpu.deformed_xp_symmetric`` and ``knowledge/deformed-xp.md``.

The verdict mirrors ``H_I``: ``H_II`` reproduces the average zeros' two leading
terms at the *same* scale ``l_x l_p = 2 pi`` (no rescaling; Sierra eq. 5.18) -- it
lifts the classical floor ``2h -> 4h`` and shifts the subleading terms, but the
``7/8`` and the GUE fluctuations are still absent. Forward, not inverse: a geometric
deformation of ``xp``; no primes, no zeros consumed.

    uv run python scripts/run_deformed_xp_symmetric.py
    uv run python scripts/run_deformed_xp_symmetric.py --heights 1000 5000 50000
"""

from __future__ import annotations

import argparse

import numpy as np

from zeta_spectral_gpu import deformed_xp_symmetric as sym


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[1000.0, 5000.0, 20000.0, 100000.0],
        help="energies E at which to evaluate the counting functions",
    )
    args = ap.parse_args()

    print(
        "Berry-Keating x<->p-symmetric deformed-xp: semiclassical count\n"
        f"  scale  l_x l_p = h = {sym.H_PRODUCT:.6g} (2*pi), same as H_I\n"
        f"  H_II classical floor 4h = {sym.classical_bound():.6g} "
        f"(H_I's is 2h); no rescaling needed to reach the average zeros.\n"
    )

    # Same scale h for both: H_II's count and H_I's share the leading two terms
    # (E/2pi)(log(E/h) - 1); they differ only at the O(1) constant / subleading, and
    # both track the average (smooth) zero count N_bar up to the missing 7/8.
    print(
        f"{'E':>9}  {'n_II':>11}  {'n_I (5.17)':>11}  "
        f"{'N_bar':>11}  {'n_II - N_bar':>13}"
    )
    for E in args.heights:
        n2 = sym.classical_count(E)
        n1 = sym.leading_count_asymmetric(E)
        nb = float(sym.average_count(E))
        print(f"{E:9.0f}  {n2:11.3f}  {n1:11.3f}  {nb:11.3f}  {n2 - nb:+13.4f}")

    big = max(args.heights)
    r = sym.classical_count(big) * sym.TWO_PI / big - np.log(big / sym.H_PRODUCT)
    print(
        f"\nleading asymptotic (E={big:.0f}): "
        f"n_II * 2pi/E - log(E/h) = {r:+.4f}  -> -1 (pins the scale, = H_I's)"
    )
    print(
        "\nverdict: restoring the x<->p symmetry leaves the mean spectral density "
        "unchanged\n         (same leading two terms as H_I, same scale); it lifts "
        "the classical floor\n         2h -> 4h and shifts the subleading terms, but "
        "the 7/8 and the GUE\n         fluctuations are still absent -- average "
        "density yes, fluctuations no (cf #24)."
    )


if __name__ == "__main__":
    main()
