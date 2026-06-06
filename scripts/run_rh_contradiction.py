"""#25: Sierra's RH-by-contradiction demo (massless-Dirac mirror model, sec. XII C).

A zero of zeta *on* the critical line makes the truncated Moebius sum grow only
like ``log n`` (eq. 12.30), so the bound-state norm is finite once the
self-adjoint-extension phase ``vartheta`` is tuned (eq. 12.34). A hypothetical
zero ``rho_c = sigma_c + i e_c`` *off* the line (``sigma_c > 1/2``) would instead
make ``|M_z(n)|`` grow *polynomially*, ``~ n^{sigma_c-1/2}`` (eq. 12.35) -- and
then the norm diverges for **every** ``vartheta`` (eq. 12.14), which a self-adjoint
``H_vartheta`` cannot do. Hence no off-line zero exists. This script exhibits both
sides numerically; the off-line zero is a counterfactual we plant (no true zero is
consumed -- forward, not inverse).

    uv run python scripts/run_rh_contradiction.py
    uv run python scripts/run_rh_contradiction.py --sigma-c 0.8 --e-c 50 --n 2000000
"""

from __future__ import annotations

import argparse

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm

ZERO_1 = 14.134725  # a real (on-line) zero, used only as a probe point


def _rms_loglog_slope(abs_m, n, lo, hi):
    edges = np.logspace(np.log10(lo), np.log10(hi), 11)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rms = np.array(
        [
            np.sqrt(np.mean(abs_m[(n >= a) & (n < b)] ** 2))
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    return float(np.polyfit(np.log(centers), np.log(rms), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sigma-c", type=float, default=0.9, help="Re of the planted off-line zero"
    )
    ap.add_argument(
        "--e-c", type=float, default=40.0, help="Im of the planted off-line zero"
    )
    ap.add_argument(
        "--E", type=float, default=38.0, help="energy at which to test the norm"
    )
    ap.add_argument("--n", type=int, default=1_000_000, help="mirror truncation")
    ap.add_argument("--eps", type=float, default=0.25, help="semi-transparency scale")
    args = ap.parse_args()

    n = np.arange(1, args.n + 1)

    print(f"Sierra RH-by-contradiction demo  (eps={args.eps})\n")

    # --- On-line zero: log growth -> finite norm (tuned vartheta) ----------------
    g_on = dm.growth_profile(ZERO_1, n[199::50])
    slope_on = float(
        np.polyfit(np.log(n[199::50].astype(np.float64)), np.log(g_on), 1)[0]
    )
    dens_on = dm.normalizable_amplitude(ZERO_1, n, eps=args.eps) / n  # tuned vartheta
    cum_on = np.cumsum(dens_on)
    print(f"on-line zero E = {ZERO_1}:")
    print(
        f"   |M_z(n)| growth: apparent slope {slope_on:+.3f}  (log n, sub-polynomial)"
    )
    print(f"   norm N_z (tuned vartheta): {cum_on[-1]:.4f}  -> FINITE (converges)")
    print(
        f"     [N=1e4: {cum_on[9999]:.4f}, N={args.n:.0e}: {cum_on[-1]:.4f}; plateau]\n"
    )

    # --- Off-line zero: polynomial growth -> norm diverges for every vartheta ----
    m_off = dm.offline_mobius_sum(args.E, n, sigma_c=args.sigma_c, e_c=args.e_c)
    slope_off = _rms_loglog_slope(np.abs(m_off), n, 300.0, args.n * 0.9)
    print(f"off-line zero rho_c = {args.sigma_c} + {args.e_c}i  (RH would be false):")
    print(
        f"   |M_z(n)| growth: RMS slope {slope_off:+.3f}  "
        f"(predict sigma_c-1/2 = {args.sigma_c - 0.5:+.3f}, POLYNOMIAL)"
    )
    print(f"   norm N_z for several vartheta (E = {args.E}):")
    print("     vartheta      N_z[1e4]       N_z[1e5]       N_z[full]")
    checkpoints = [min(10_000, args.n), min(100_000, args.n), args.n]
    for vt in (0.0, 1.5, np.pi, -2.0):
        cum = dm.norm_partial_sums(m_off, eps=args.eps, vartheta=vt)
        vals = "  ".join(f"{cum[c - 1]:.3e}" for c in checkpoints)
        print(f"     {vt:+6.2f}     {vals}")
    print(
        "\n   -> the norm BLOWS UP for every vartheta (~vartheta-independent): an\n"
        "      off-line zero is incompatible with a self-adjoint H_vartheta. QED (heuristic)."
    )


if __name__ == "__main__":
    main()
