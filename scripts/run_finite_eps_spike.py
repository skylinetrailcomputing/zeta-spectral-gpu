"""#45 spike: does a finite-eps operator-intrinsic spectral object exist? (gate for #44)

Sierra's prime-driven model (#25) lives entirely in the eps -> 0 semiclassical
limit, where the transfer-matrix product collapses (BCH) to the Moebius locator
``M'_z(n)``. #44 asks whether the **exact** finite-eps product ``T_n...T_2``
(eq. 115) carries an operator-intrinsic, non-circular spectral density / resonance
structure one could form-factor. This script computes the evidence for the
GO/NO-GO ruling (``_private/issue-44-resonance-ruling.md``):

  1. **Harmonic model (Appendix A)** -- the *only* model with a genuine finite-eps
     spectrum (continuum bands + discrete levels). The exact-product code
     reproduces eq. A16 / A13. But its periods are integers (eq. 67): NO primes.
  2. **Moebius (Riemann) model** -- at finite eps the exact product equals the
     eps -> 0 locator up to O(eps^3) corrections, and carries no intrinsic
     discrete spectrum: a bound state appears only with the per-zero ``vartheta``
     tuning (Piece B, inverse). The only structure is the locator's zero-peaks
     (peak-finding = the #25 Phase-3 circular trap).

    uv run python scripts/run_finite_eps_spike.py
"""

from __future__ import annotations

import argparse

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm

ZERO_1 = 14.134725  # a real zero, used only as a probe point (never fed in)
NONZERO = 17.0


def harmonic_section(eps: float) -> None:
    print("=" * 70)
    print(f"1. HARMONIC MODEL (App. A) -- finite-eps spectrum exists, eps={eps}")
    print("=" * 70)
    delta = dm.harmonic_gap_half_width(eps)
    print(
        f"  band half-gap delta = {delta:.6f};  "
        f"sin(pi delta) = {np.sin(np.pi * delta):.6f}  vs  "
        f"2eps/(1+eps^2) = {2 * eps / (1 + eps**2):.6f}   (eq. A16)"
    )
    # scan continuum vs analytic band edges
    Es = np.linspace(2 * np.pi, 4 * np.pi, 200001)
    cont = dm.harmonic_is_continuum(Es, eps)
    lo, hi = Es[np.argmax(cont)], Es[len(cont) - 1 - np.argmax(cont[::-1])]
    b_lo, b_hi = dm.harmonic_bands(eps, [1])[0]
    print(
        f"  band m=1: scanned [{lo:.4f}, {hi:.4f}]  analytic [{b_lo:.4f}, {b_hi:.4f}]"
    )
    # exact product: band bounded, gap grows, discrete level decays
    e_band, e_gap = 2 * np.pi * 1.5, 2 * np.pi * 2 - 0.1
    nb = dm.harmonic_amplitude_norms(np.array([e_band, e_gap]), 200, eps=eps)
    nd = dm.harmonic_amplitude_norms(2 * np.pi, 40, eps=eps, vartheta=0.0)[0]
    print(
        f"  exact <A_k|A_k>:  band E={e_band:.2f} -> max {nb[0].max():.2f} (BOUNDED, "
        f"continuum)"
    )
    print(f"                    gap  E={e_gap:.2f} -> {nb[1, -1]:.2e} (GROWS)")
    print(
        f"  discrete E=2pi (vartheta=0): <A_k|A_k> {nd[0]:.1f} -> {nd[20]:.1e} -> "
        f"{nd[39]:.1e}  (BOUND STATE)"
    )
    print("  => a real finite-eps density of states -- but periods are INTEGERS.\n")


def mobius_section(eps_list: list[float], n: int) -> None:
    print("=" * 70)
    print("2. MOEBIUS (RIEMANN) MODEL -- finite eps adds nothing operator-intrinsic")
    print("=" * 70)

    # (a) exact product collapses to the BCH locator as eps -> 0
    print("  (a) exact product vs eps->0 BCH locator (max|exact-BCH| over k<=120):")
    print("        E          " + "   ".join(f"eps={e:g}" for e in eps_list))
    for E in (ZERO_1, NONZERO):
        row = []
        for eps in eps_list:
            ex = dm.mobius_amplitude_norms(E, 120, eps=eps, vartheta=0.3)[0]
            bc = dm.mobius_amplitude_norms(E, 120, eps=eps, vartheta=0.3, bch=True)[0]
            row.append(np.max(np.abs(ex - bc)[2:]))
        tag = "zero" if E == ZERO_1 else "non-zero"
        print(f"     {E:6.2f} ({tag:8}) " + "  ".join(f"{d:.2e}" for d in row))
    slope = np.polyfit(
        np.log(eps_list),
        np.log(
            [
                np.max(
                    np.abs(
                        dm.mobius_amplitude_norms(ZERO_1, 120, eps=e, vartheta=0.3)[0]
                        - dm.mobius_amplitude_norms(
                            ZERO_1, 120, eps=e, vartheta=0.3, bch=True
                        )[0]
                    )[2:]
                )
                for e in eps_list
            ]
        ),
        1,
    )[0]
    print(f"     => log-log slope ~ {slope:.2f}  (vanishes faster than eps^2)\n")

    # (b) no intrinsic discrete spectrum: continuum at finite eps
    print(f"  (b) finite-eps spectrum is a CONTINUUM (eps={eps_list[0]}, n={n}):")
    for E, tag in ((NONZERO, "non-zero"), (ZERO_1, "zero")):
        gen = dm.mobius_amplitude_norms(E, n, eps=eps_list[0], vartheta=0.0)[0]
        print(
            f"     E={E:6.2f} ({tag:8}) generic vartheta=0: <A_k|A_k> tail mean "
            f"{gen[-500:].mean():.3f}  (no decay -> not a discrete eigenstate)"
        )
    tuned = dm.mobius_amplitude_norms(
        ZERO_1, n, eps=eps_list[0], vartheta=dm.riemann_tuning_phase(ZERO_1)
    )[0]
    print(
        f"     E={ZERO_1:6.2f} (zero)     TUNED vartheta(E):  tail mean "
        f"{tuned[-500:].mean():.3f}  (decays -> bound state, but vartheta uses "
        "theta(E) = Piece B, INVERSE)"
    )
    print(
        "  => the only forward output is the locator's zero-peaks; reading a\n"
        "     spectrum off it = peak-finding the known zeros (circular). NO-GO.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eps", type=float, default=0.2, help="harmonic coupling")
    ap.add_argument("--n", type=int, default=4000, help="Moebius mirror truncation")
    args = ap.parse_args()
    harmonic_section(args.eps)
    mobius_section([0.08, 0.04, 0.02], args.n)


if __name__ == "__main__":
    main()
