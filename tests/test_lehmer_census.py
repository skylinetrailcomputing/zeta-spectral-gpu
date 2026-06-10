"""Lehmer-pair census pipeline vs mpmath / literature ground truth (issue #86).

Forward throughout: every zero consumed by the census is produced by the
Riemann-Siegel scan itself. ``mp.zetazero`` and the published Lehmer-pair data
(the classical pair near t ~ 7005; the COSV 1993 bound) appear only as
after-the-fact checks, never as input.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from zeta_spectral_gpu import lehmer_census as lc


def test_scan_zeros_matches_mpmath_zetazero():
    # Scan a low window and check every located ordinate against mp.zetazero
    # (the truth), starting from the index the smooth count predicts.
    w = lc.scan_zeros(100.0, 140.0)
    assert abs(w.zeros.size - w.expected_count) < 2.0
    # N(100) = 29 (gamma_29 ~ 98.83): the first zero above 100 is gamma_30.
    for i, gamma in enumerate(w.zeros):
        truth = float(mp.zetazero(30 + i).imag)
        assert abs(gamma - truth) < 2e-6, (i, gamma, truth)


def test_count_check_tracks_theta():
    w = lc.scan_zeros(1000.0, 1100.0)
    assert abs(w.zeros.size - w.expected_count) < 2.0


def test_normalized_gaps_have_unit_mean():
    w = lc.scan_zeros(1000.0, 1300.0)
    s = lc.normalized_gaps(w.zeros)
    assert abs(float(s.mean()) - 1.0) < 0.03
    assert np.all(s > 0.0)


def test_dip_rescan_recovers_close_pair_missed_by_coarse_grid():
    # With a deliberately coarse grid (step ~ 0.27, gap ~ 0.038) the classical
    # Lehmer pair falls inside one cell -- both sign changes invisible to the
    # coarse scan -- and must be recovered by the |Z|-dip rescan tier.
    w = lc.scan_zeros(7000.0, 7010.0, step_fraction=0.3)
    for truth in (7005.06287, 7005.10056):
        assert np.min(np.abs(w.zeros - truth)) < 1e-4, truth
    assert w.rescued.size >= 2


def test_census_finds_classical_lehmer_pair():
    # End-to-end: scan around t ~ 7005, census the gaps, and the famous
    # Lehmer pair (gamma_6709, gamma_6710) must come out as the top-quality
    # row, classified as a CSV Lehmer pair. Its ordinates are then scored
    # against mp.zetazero -- the zeros never entered the pipeline.
    w = lc.scan_zeros(6900.0, 7110.0)
    rows = lc.lehmer_census(w)
    assert rows, "census found no candidate pairs"
    best = rows[0]
    assert abs(best.gamma_minus - 7005.0627) < 1e-3
    assert best.delta2g < 0.05  # spectacular quality: far under the 4/5 bar
    assert best.lam is not None
    assert -8e-4 < best.lam < -6e-4  # lambda ~ -Delta^2/2 (classical units)
    truth_m = float(mp.zetazero(6709).imag)
    truth_p = float(mp.zetazero(6710).imag)
    assert abs(best.gamma_minus - truth_m) < 2e-5
    assert abs(best.gamma_plus - truth_p) < 2e-5
    # every Lehmer-classified row carries a negative lambda above -1/(8g)*4
    for r in rows:
        if r.lam is not None:
            assert r.lam < 0.0
            g = r.delta2g / (r.gamma_plus - r.gamma_minus) ** 2
            assert r.lam > -4.0 / (8.0 * g)


def test_csv_lambda_pins_cosv_1993_bound():
    # The COSV 1993 pair (Delta ~ 1.0857e-4 at t ~ 3.8886e8, reconstructed
    # from Stopple's section-6 data) published the classical bound -5.895e-9.
    # csv_lambda must reproduce it -- and be insensitive to g at this quality
    # (lambda ~ -Delta^2/2 + O(Delta^4 g)), which pins the x4 normalization.
    delta = 1.0857e-4
    for g in (5.0, 53.0, 200.0):
        lam = lc.csv_lambda(delta, g)
        assert abs(lam - (-5.895e-9)) < 0.005e-9, (g, lam)
    # exact small-quality asymptote
    assert abs(lc.csv_lambda(1e-6, 10.0) - (-0.5e-12)) < 1e-15


def test_csv_lambda_rejects_non_lehmer_pair():
    with pytest.raises(ValueError):
        lc.csv_lambda(1.0, 1.0)  # Delta^2 g = 1 >= 4/5


def test_density_integral_matches_quadrature():
    rho = lambda x: np.log(x / lc.TWO_PI) / lc.TWO_PI  # noqa: E731

    for a, b, c in [
        (lc.TWO_PI, 6900.0, 7005.0),  # below-window tail
        (7110.0, 5.0e5, 7005.0),  # above-window tail (finite)
        (lc.TWO_PI, 5.0e5, -7005.0),  # mirror axis
    ]:
        x = np.linspace(a, b, 400_001)
        quad = float(np.trapezoid(rho(x) / (x - c) ** 2, x))
        closed = lc._density_integral(a, b, c)
        assert abs(closed - quad) < 1e-4 * abs(quad) + 1e-12, (a, b, c)
    # the infinite upper tail equals the finite one plus a remainder that
    # vanishes as the cut grows
    near = lc._density_integral(7110.0, 1.0e7, 7005.0)
    full = lc._density_integral(7110.0, np.inf, 7005.0)
    assert full > near
    assert full - near < 1e-5


def test_density_integral_rejects_interior_pole():
    with pytest.raises(ValueError):
        lc._density_integral(100.0, 200.0, 150.0)


def test_csv_g_dominated_by_censused_zeros():
    # For the classical pair the analytic tails are a per-mille correction:
    # g from the censused window alone and g with tails agree closely, and
    # the tails only add (all terms positive).
    w = lc.scan_zeros(6900.0, 7110.0)
    i = int(np.argmin(np.abs(w.zeros - 7005.0627)))
    x = w.zeros
    gm, gp = x[i], x[i + 1]
    others = np.delete(x, [i, i + 1])
    g_window = float(np.sum(1.0 / (others - gm) ** 2 + 1.0 / (others - gp) ** 2))
    g_full = lc.csv_g(x, i, t_lo=w.t_lo, t_hi=w.t_hi)
    assert g_full > g_window
    assert (g_full - g_window) / g_full < 0.05


def test_gap_resolution_floor_budget():
    # The budget the issue asked to write down: the floor grows with height
    # but stays orders of magnitude below the gaps a feasible census contains.
    s6 = lc.gap_resolution_floor(1.0e6)
    s8 = lc.gap_resolution_floor(1.0e8)
    assert s6 < s8
    assert 5e-5 < s6 < 1e-3
    assert s8 < 5e-3
    # larger amplitude scale (measured RMS) lowers the floor
    assert lc.gap_resolution_floor(1.0e6, z_scale=4.0) < s6


def test_polish_pair_refines_fp64_zeros():
    w = lc.scan_zeros(7000.0, 7010.0)
    i = int(np.argmin(np.abs(w.zeros - 7005.0627)))
    gm, gp = lc.polish_pair(w.zeros[i], w.zeros[i + 1], dps=25)
    assert abs(gm - float(mp.zetazero(6709).imag)) < 1e-9
    assert abs(gp - float(mp.zetazero(6710).imag)) < 1e-9
    # polish moved the fp64 estimates by less than the local resolution budget
    assert abs(gm - w.zeros[i]) < 1e-5
    assert abs(gp - w.zeros[i + 1]) < 1e-5


def test_gue_small_gap_laws():
    s = np.array([0.01, 0.05, 0.1])
    np.testing.assert_allclose(
        lc.gue_small_gap_density(s), (np.pi**2 / 3.0) * s**2, rtol=1e-12
    )
    # CDF is the integral of the density
    grid = np.linspace(0.0, 0.1, 10_001)
    quad = np.trapezoid(lc.gue_small_gap_density(grid), grid)
    assert abs(quad - float(lc.gue_small_gap_cdf(0.1))) < 1e-8
