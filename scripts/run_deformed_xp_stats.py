"""Warm-up #24: universality statistics of the deformed-xp spectrum.

The statistics payoff on top of #23. Takes Sierra's deformed-``xp`` spectrum (the
real roots of the Bessel-K secular equation; ``deformed_xp.secular_spectrum``) and
runs it through the *same* warm-up harness the Riemann zeros go through — nearest-
neighbour spacing ``P(s)`` (#5), number variance ``Sigma^2(L)`` and the Dyson-Mehta
``Delta_3(L)`` (#15) — overlaying both against the GUE and Poisson references.

The expected answer is a *known* one, and that is the point: the deformed-``xp``
operator reproduces the **average** zeros, so its spectrum unfolds to a near-picket
fence — ``P(s) -> delta(s-1)``, ``Sigma^2(L)`` bounded, ``Delta_3(L)`` flat — with
no trace of the GUE fluctuations the real zeros carry (Sierra & Rodríguez-Laguna
2011, arXiv:1102.5356, eqs. 14–16; Sierra, *Symmetry* 11(4) 494, 2019,
arXiv:1601.01797 §V). A non-GUE result is the finding, not a failure: matching the
mean density (which the xp model nails) is necessary but nowhere near sufficient —
exactly the bar the CCM flagship (#9/#16) must clear at the *fluctuation* level.

Forward, not inverse (see README / project-framing): the #23 operator consumes
neither primes nor zeros; the zeros appear here only as an output comparison
(overlaid curves), never as input. Nothing is fitted.

    uv run python scripts/run_deformed_xp_stats.py
    uv run python scripts/run_deformed_xp_stats.py --n 300 --source first-100k
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _gue_spacing_var() -> float:
    """Variance of the GUE Wigner surmise (numeric reference, ~0.178)."""
    s = np.linspace(0.0, 12.0, 20001)
    p = spacing.gue_wigner_surmise(s)
    mean = np.trapezoid(s * p, s)
    return float(np.trapezoid((s - mean) ** 2 * p, s))


def _rigidity(unfolded, lengths, n_offsets, use_gpu):
    """``Sigma^2(L)`` and ``Delta_3(L)`` on the GPU when available, else CPU."""
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            return (
                spacing_gpu.number_variance_gpu(unfolded, lengths, n_offsets=n_offsets),
                spacing_gpu.dyson_mehta_delta3_gpu(
                    unfolded, lengths, n_offsets=n_offsets
                ),
                "gpu",
            )
        except ImportError:
            pass
    return (
        spacing.number_variance(unfolded, lengths, n_offsets=n_offsets),
        spacing.dyson_mehta_delta3(unfolded, lengths, n_offsets=n_offsets),
        "cpu",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=160, help="deformed-xp eigenvalues")
    ap.add_argument(
        "--source",
        default="mpmath",
        help="real-zero source: 'mpmath' (default, matched count) or an Odlyzko table",
    )
    ap.add_argument(
        "--zeros-n",
        type=int,
        default=None,
        help="real zeros to compare (default: match --n for a fair sample size)",
    )
    ap.add_argument(
        "--theta",
        type=float,
        default=dxp.THETA_RIEMANN,
        help="self-adjoint extension angle (default pi/4, the Riemann case)",
    )
    ap.add_argument("--dps", type=int, default=None, help="mpmath precision override")
    ap.add_argument("--l-min", type=float, default=1.0)
    ap.add_argument("--l-max", type=float, default=12.0)
    ap.add_argument("--n-l", type=int, default=24, help="number of window lengths")
    ap.add_argument("--n-offsets", type=int, default=4000)
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--s-max", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=DATA / "deformed_xp_stats.png")
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU path")
    ap.add_argument("--refresh", action="store_true", help="ignore the cached spectrum")
    args = ap.parse_args()

    zeros_n = args.zeros_n or args.n

    # The two spectra, both unfolded to unit mean: the deformed-xp levels by the
    # analytic smooth count (exact for this model), the zeros empirically the same.
    t0 = time.perf_counter()
    spec = dxp.secular_spectrum(
        args.n, theta=args.theta, dps=args.dps, refresh=args.refresh
    )
    w_xp = dxp.analytic_unfold(spec)
    dps = args.dps or dxp.required_dps(args.n)
    print(
        f"deformed-xp: {spec.size} levels in [{spec[0]:.2f}, {spec[-1]:.2f}]  "
        f"(theta={args.theta / np.pi:.4g}*pi, dps={dps}, "
        f"{time.perf_counter() - t0:.0f}s)"
    )

    tau = zeros.load_ordinates(zeros_n, source=args.source, cache_dir=DATA)
    w_z = zeros.unfold(tau)
    print(f"real zeros : {tau.size} ordinates from {args.source!r}")

    lengths = np.geomspace(args.l_min, args.l_max, args.n_l)

    s_xp = spacing.nearest_neighbour_spacings(w_xp)
    s_z = spacing.nearest_neighbour_spacings(w_z)
    sig_xp, d3_xp, backend = _rigidity(w_xp, lengths, args.n_offsets, not args.no_gpu)
    sig_z, d3_z, _ = _rigidity(w_z, lengths, args.n_offsets, not args.no_gpu)

    # The discrimination read-out: picket fence (xp) vs GUE (zeros).
    lmax = float(lengths[-1])
    gue_sig = float(spacing.gue_number_variance(lmax))
    print(f"\nbackend={backend}   L in [{args.l_min:g}, {lmax:g}] ({args.n_l} pts)")
    print("                      deformed-xp      real zeros        GUE ref")
    print(
        f"  var(s) [P(s)->1]    {s_xp.var():10.4f}      {s_z.var():10.4f}     "
        f"{_gue_spacing_var():10.4f}"
    )
    print(
        f"  Sigma^2(L_max)      {sig_xp[-1]:10.4f}      {sig_z[-1]:10.4f}     "
        f"{gue_sig:10.4f}"
    )
    print(
        f"  Delta_3(L_max)      {d3_xp[-1]:10.4f}      {d3_z[-1]:10.4f}     "
        f"{float(spacing.gue_delta3(lmax)):10.4f}"
    )

    # The xp spacings cluster at 1 (var << GUE) and its rigidity stays well below
    # the GUE logarithm — the harness must separate the two or it is broken.
    discriminates = s_xp.var() < 0.25 * s_z.var() and np.nanmax(sig_xp) < 0.5 * gue_sig
    print(
        "\nverdict: the harness "
        + ("DISCRIMINATES" if discriminates else "FAILS to separate")
        + " the picket-fence xp spectrum from the GUE\n         statistics of the "
        "real zeros - matching the mean density is not enough."
    )

    out = plots.deformed_xp_stats_figure(
        spacings_xp=s_xp,
        spacings_zeros=s_z,
        lengths=lengths,
        sigma2_xp=sig_xp,
        sigma2_zeros=sig_z,
        delta3_xp=d3_xp,
        delta3_zeros=d3_z,
        n_xp=spec.size,
        n_zeros=tau.size,
        out_path=args.out,
        n_bins=args.n_bins,
        s_max=args.s_max,
    )
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
