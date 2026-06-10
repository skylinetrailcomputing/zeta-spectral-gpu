"""Invariants for the arithmetic-correlations module (issue #84).

All offline and zero-free: the predictions are pure prime/zeta-side objects, so
they are pinned by mathematical identities — above all the exact equality of
the Conrey–Snaith Theorem 4.1 integrand and the Bogomolny–Keating
Hardy–Littlewood form, two independently transcribed published formulas. The
empirical Fourier statistic is checked against a direct loop on synthetic data;
the real-zero science lives in ``scripts/run_arithmetic_correlations.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import arithmetic_correlations as ac

TWO_PI = 2.0 * np.pi


# --- Prime-side inputs -------------------------------------------------------


def test_primes_upto_small():
    np.testing.assert_array_equal(
        ac.primes_upto(30), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    )
    assert ac.primes_upto(1).size == 0
    assert ac.primes_upto(2).tolist() == [2]


def test_von_mangoldt_values():
    lam = ac.von_mangoldt(16)
    assert lam[0] == lam[1] == 0.0
    assert lam[2] == pytest.approx(np.log(2))
    assert lam[4] == pytest.approx(np.log(2))  # prime power, weight log p
    assert lam[8] == pytest.approx(np.log(2))
    assert lam[16] == pytest.approx(np.log(2))
    assert lam[9] == pytest.approx(np.log(3))
    assert lam[6] == lam[12] == lam[15] == 0.0  # composite, not prime power
    assert lam[13] == pytest.approx(np.log(13))


# --- The BK == CS identity ---------------------------------------------------


def test_bk_equals_cs_exactly():
    """The two published forms of the arithmetic pair correlation are one
    formula: CS's A/B regrouped into BK's Phi_diag/Phi_off (same prime cutoff
    on both sides). Agreement far below any plotting scale pins both
    transcriptions at once.
    """
    eps = np.linspace(0.1, 5.0, 12)
    t = 1.0e5
    cs = ac.cs_integrand(eps, t, p_max=2000)
    bk = ac.bk_integrand(eps, t, p_max=2000)
    np.testing.assert_allclose(bk, cs, rtol=0, atol=1e-10)


def test_cs_unfolds_to_gue_at_large_height():
    """Arithmetic terms are O(1/log t): at t = 1e12 the unfolded CS density
    must sit on the sine kernel to a couple of percent."""
    t = 1.0e12
    dbar = np.log(t / TWO_PI) / TWO_PI
    x = np.linspace(0.25, 3.0, 8)  # unfolded separations
    rho = ac.cs_integrand(x / dbar, t, p_max=20_000) / dbar**2
    np.testing.assert_allclose(rho, 1.0 - np.sinc(x) ** 2, rtol=0, atol=0.02)


def test_cs_pair_density_is_the_window_integral():
    """The closed-form t-integration over a thin window reduces to the
    pointwise integrand times the window length."""
    eps = np.linspace(0.3, 4.0, 7)
    t, half = 5.0e4, 50.0
    pointwise = ac.cs_integrand(eps, t, p_max=2000)
    integrated = ac.cs_pair_density(eps, t - half, t + half, p_max=2000)
    np.testing.assert_allclose(integrated / (2 * half), pointwise, rtol=1e-5)


def test_gue_pair_density_plateau_and_repulsion():
    t_lo, t_hi = 4.0e4, 6.0e4
    dbar = np.log(5.0e4 / TWO_PI) / TWO_PI
    # Zero separation: full level repulsion, density 0.
    assert ac.gue_pair_density(1e-9, t_lo, t_hi)[0] == pytest.approx(0.0, abs=1e-6)
    # Far separation: the decorrelated plateau is exactly int dbar(t)^2 dt.
    t = np.linspace(t_lo, t_hi, 100_001)
    plateau = float(np.trapezoid((np.log(t / TWO_PI) / TWO_PI) ** 2, t))
    far = ac.gue_pair_density(500.0 / dbar, t_lo, t_hi)[0]
    assert far == pytest.approx(plateau, rel=1e-5)


# --- Window transforms -------------------------------------------------------


@pytest.mark.parametrize("kind", ["rect", "hann"])
def test_window_transform_matches_quadrature(kind):
    t_lo, t_hi = 10.0, 14.0
    y = np.array([0.0, 0.3, -0.7, 2.0, 9.0])
    t = np.linspace(t_lo, t_hi, 20_001)
    w = ac.window_weights(t, t_lo, t_hi, kind)
    numeric = np.trapezoid(
        w[None, :] * np.exp(-1j * y[:, None] * t[None, :]), t, axis=1
    )
    np.testing.assert_allclose(
        ac.window_transform(y, t_lo, t_hi, kind), numeric, rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("kind", ["rect", "hann"])
def test_window_l2_matches_quadrature(kind):
    t_lo, t_hi = 3.0, 11.0
    t = np.linspace(t_lo, t_hi, 100_001)
    w = ac.window_weights(t, t_lo, t_hi, kind)
    assert ac.window_l2(t_lo, t_hi, kind) == pytest.approx(
        float(np.trapezoid(w * w, t)), rel=1e-6
    )


def test_unknown_window_raises():
    with pytest.raises(ValueError, match="unknown window"):
        ac.window_weights(np.array([1.0]), 0.0, 2.0, "kaiser")
    with pytest.raises(ValueError, match="unknown window"):
        ac.window_transform(0.0, 0.0, 2.0, "kaiser")
    with pytest.raises(ValueError, match="unknown window"):
        ac.window_l2(0.0, 2.0, "kaiser")


# --- Empirical Fourier statistic ---------------------------------------------


def test_zero_fourier_matches_direct_sum():
    rng = np.random.default_rng(84)
    tau = np.sort(rng.uniform(100.0, 300.0, 400))
    u = np.array([0.0, 0.5, 1.3, 2.0])
    for kind in ("rect", "hann"):
        w = ac.window_weights(tau, 120.0, 280.0, kind)
        direct = np.array(
            [np.sum(w * np.exp(1j * uu * tau)) for uu in u], dtype=np.complex128
        )
        got = ac.zero_fourier(tau, u, 120.0, 280.0, window=kind, chunk=128)
        np.testing.assert_allclose(got, direct, rtol=0, atol=1e-10)


def test_zero_fourier_at_zero_frequency_counts_weights():
    tau = np.linspace(50.0, 150.0, 1001)
    s0 = ac.zero_fourier(tau, 0.0, 60.0, 140.0, window="rect")[0]
    n_inside = int(np.sum((tau >= 60.0) & (tau <= 140.0)))
    assert s0 == pytest.approx(n_inside)


def test_prime_prediction_isolated_peak_height():
    """At u = log 2 with a rect window, the explicit-formula prediction is the
    Landau peak -(dT/2pi) Lambda(2)/sqrt(2) up to tiny sidelobes of the other
    prime powers (and no zeros are consumed anywhere)."""
    t_lo, t_hi = 1000.0, 3000.0
    pred = ac.prime_prediction(np.array([np.log(2.0)]), t_lo, t_hi, window="rect")
    expected = (t_hi - t_lo) / TWO_PI * np.log(2.0) / np.sqrt(2.0)
    assert np.abs(pred[0]) == pytest.approx(expected, rel=0.01)


def test_diagonal_ramp_slope_and_plateau():
    t_lo, t_hi = 4.0e4, 6.0e4
    dbar = np.log(5.0e4 / TWO_PI) / TWO_PI
    for kind in ("rect", "hann"):
        l2 = ac.window_l2(t_lo, t_hi, kind)
        ramp = ac.diagonal_ramp(np.array([1.0]), t_lo, t_hi, window=kind)[0]
        assert ramp == pytest.approx(l2 / TWO_PI, rel=1e-12)
        plateau = ac.diagonal_ramp(
            np.array([10.0 * TWO_PI * dbar]), t_lo, t_hi, window=kind
        )[0]
        assert plateau == pytest.approx(dbar * l2, rel=1e-12)
