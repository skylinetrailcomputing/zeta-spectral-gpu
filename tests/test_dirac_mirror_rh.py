"""RH-by-contradiction demo for the prime-driven mirror model (#25, Sierra XII C).

Sierra's heuristic "proof" of RH: a zero ``rho_c = sigma_c + i e_c`` off the
critical line (``sigma_c > 1/2``) would make the truncated Moebius sum grow
*polynomially*, ``|M_z(x)| ~ x^{sigma_c-1/2}`` (eq. 12.35), instead of the ``log x``
of an on-line zero (eq. 12.30). That fast growth makes the eigenstate
non-normalizable for **every** self-adjoint-extension phase ``vartheta``
(eq. 12.14), which is impossible for the self-adjoint ``H_vartheta`` -- so no
off-line zero can exist. These tests pin the growth law and the
``vartheta``-independent norm divergence. Forward: the off-line zero is a
*counterfactual* plugged in to probe the structure -- no true zero is consumed.
"""

from __future__ import annotations

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm

ZERO_1 = 14.134725  # a real (on-line) zero, used only as a probe point


def _rms_loglog_slope(abs_m: np.ndarray, n: np.ndarray, lo: float, hi: float) -> float:
    """Slope of log(RMS|M|) vs log n in log-spaced bins (averages out the cos)."""
    edges = np.logspace(np.log10(lo), np.log10(hi), 11)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rms = np.array(
        [
            np.sqrt(np.mean(abs_m[(n >= a) & (n < b)] ** 2))
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    return float(np.polyfit(np.log(centers), np.log(rms), 1)[0])


def test_offline_zero_grows_with_predicted_exponent():
    # eq. 12.35: a hypothetical zero at sigma_c + i e_c gives |M_z(n)| ~ n^{sigma_c-1/2}.
    n = np.arange(200, 200001)
    for sigma_c in (0.7, 0.8, 0.9):
        m = dm.offline_mobius_sum(20.0, n, sigma_c=sigma_c, e_c=38.0)
        slope = _rms_loglog_slope(np.abs(m), n, 300.0, 180000.0)
        assert abs(slope - (sigma_c - 0.5)) < 0.03


def test_online_zero_growth_is_sub_polynomial():
    # eq. 12.30: at a real zero |M_z(n)| ~ log n, so the apparent power-law slope
    # is near zero -- far below the n^{sigma_c-1/2} (>= 0.2) of any off-line zero.
    n = np.arange(200, 200001, 50)
    abs_m = dm.growth_profile(ZERO_1, n)
    slope = float(np.polyfit(np.log(n.astype(np.float64)), np.log(abs_m), 1)[0])
    assert slope < 0.15


def test_offline_norm_diverges_for_every_vartheta():
    # The norm blows up regardless of the self-adjoint-extension phase vartheta
    # (Phi_z keeps oscillating, so no vartheta cancels the growing mode):
    # contradiction with self-adjointness => no off-line zero. sigma_c=0.9, near e_c.
    m = dm.offline_mobius_sum(38.0, np.arange(1, 50001), sigma_c=0.9, e_c=40.0)
    finals = [
        dm.norm_partial_sums(m, eps=0.25, vartheta=vt)[-1]
        for vt in (0.0, 1.5, np.pi, -2.0)
    ]
    assert all(f > 1e6 for f in finals)  # diverges (huge) for every vartheta
    assert max(finals) / min(finals) < 50.0  # ... and ~independent of vartheta


def test_online_zero_norm_converges_with_tuned_vartheta():
    # Contrast: at a real zero with vartheta tuned to eq. 12.33 the norm density
    # decays and the partial sums plateau (eq. 12.34, finite). Norm density is the
    # (1/n)<A_n|A_n> summand of eq. 12.1/12.14.
    n_max = 40000
    tr = np.arange(1, n_max + 1)
    dens = dm.normalizable_amplitude(ZERO_1, tr, eps=0.25) / tr  # tuned vartheta
    cum = np.cumsum(dens)
    tail = cum[-1] - cum[n_max // 2]
    assert tail < 0.05 * cum[-1]  # last half adds almost nothing -> converging
