"""Invariants for the spacing statistics, including GPU-vs-CPU agreement.

GPU tests skip cleanly when cupy isn't installed, so the CPU invariants still run
on a bare environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from zeta_spectral_gpu import spacing


def test_surmises_are_normalised():
    for fn in (
        spacing.gue_wigner_surmise,
        spacing.goe_wigner_surmise,
        spacing.poisson_surmise,
    ):
        total, _ = integrate.quad(fn, 0.0, np.inf)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_surmises_have_unit_mean():
    for fn in (
        spacing.gue_wigner_surmise,
        spacing.goe_wigner_surmise,
        spacing.poisson_surmise,
    ):
        mean, _ = integrate.quad(lambda s, f=fn: s * f(s), 0.0, np.inf)
        assert mean == pytest.approx(1.0, abs=1e-6)


def test_nearest_neighbour_matches_diff():
    x = np.cumsum(np.abs(np.sin(np.arange(100))) + 0.1)
    np.testing.assert_allclose(spacing.nearest_neighbour_spacings(x), np.diff(x))


def test_montgomery_pair_correlation_known_points():
    # R2(0) = 0 (level repulsion); R2(1) = 1 (sin pi = 0); R2(0.5) = 1 - (2/pi)^2.
    assert spacing.montgomery_pair_correlation(0.0) == pytest.approx(0.0, abs=1e-12)
    assert spacing.montgomery_pair_correlation(1.0) == pytest.approx(1.0, abs=1e-12)
    assert spacing.montgomery_pair_correlation(0.5) == pytest.approx(
        1.0 - (2.0 / np.pi) ** 2
    )
    # decorrelates: R2 -> 1 for large r.
    assert spacing.montgomery_pair_correlation(50.0) == pytest.approx(1.0, abs=2e-3)


def test_pair_correlation_density_flat_hist_is_unit():
    # A flat forward histogram of n_levels * bin_width per bin normalises to R2 == 1.
    bin_width, n_levels, n_bins = 0.05, 10_000, 60
    hist = np.full(n_bins, n_levels * bin_width)
    centres, r2 = spacing.pair_correlation_density(hist, bin_width, n_levels)
    np.testing.assert_allclose(r2, 1.0)
    np.testing.assert_allclose(centres[0], 0.5 * bin_width)


def test_pair_correlation_density_recovers_montgomery():
    # Build a hist whose counts equal n*w*R2(r); the normaliser must invert it.
    bin_width, n_levels, n_bins = 0.05, 100_000, 60
    centres = (np.arange(n_bins) + 0.5) * bin_width
    hist = n_levels * bin_width * spacing.montgomery_pair_correlation(centres)
    _, r2 = spacing.pair_correlation_density(hist, bin_width, n_levels)
    np.testing.assert_allclose(r2, spacing.montgomery_pair_correlation(centres))


def _sorted_levels(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, n, size=n))


def test_gpu_pair_correlation_matches_cpu():
    cp = pytest.importorskip("cupy")  # noqa: F841
    from zeta_spectral_gpu import spacing_gpu

    x = _sorted_levels()
    cpu = spacing.pair_correlation_histogram(x, bin_width=0.1, max_sep=5.0)
    gpu = spacing_gpu.pair_correlation_histogram_gpu(x, bin_width=0.1, max_sep=5.0)
    np.testing.assert_array_equal(cpu, gpu)


def test_gpu_spacings_match_cpu():
    cp = pytest.importorskip("cupy")  # noqa: F841
    from zeta_spectral_gpu import spacing_gpu

    x = _sorted_levels()
    cpu = spacing.nearest_neighbour_spacings(x)
    gpu = spacing_gpu.nearest_neighbour_spacings_gpu(x)
    np.testing.assert_allclose(cpu, gpu, rtol=0, atol=1e-12)


# --- Rigidity: number variance and Dyson-Mehta Delta_3 ----------------------


def test_delta3_transform_poisson_is_L_over_15():
    # Verifies the Mehta kernel (L^3 - 2L^2 r + r^3): Sigma^2(r)=r => Delta_3=L/15.
    lengths = np.array([2.0, 5.0, 13.0, 30.0])
    d3 = spacing.delta3_from_sigma2(lambda r: r, lengths)
    np.testing.assert_allclose(d3, lengths / 15.0, rtol=1e-6)


def test_gue_number_variance_limits():
    # Sigma^2 -> L as L -> 0 (small windows hold at most one level).
    small = np.array([1e-4, 1e-3])
    np.testing.assert_allclose(spacing.gue_number_variance(small), small, rtol=3e-3)
    assert float(spacing.gue_number_variance(0.0)) == 0.0
    # Sigma^2 -> (1/pi^2)(ln 2pi L + gamma + 1) as L -> infinity.
    large = np.array([100.0, 1000.0])
    asymptote = (np.log(2 * np.pi * large) + np.euler_gamma + 1.0) / np.pi**2
    np.testing.assert_allclose(spacing.gue_number_variance(large), asymptote, atol=1e-3)


def test_gue_delta3_asymptote():
    # Delta_3 -> (1/2pi^2)(ln 2pi L + gamma - 5/4): half the slope of Sigma^2.
    large = np.array([100.0, 1000.0])
    asymptote = (np.log(2 * np.pi * large) + np.euler_gamma - 1.25) / (2 * np.pi**2)
    np.testing.assert_allclose(spacing.gue_delta3(large), asymptote, atol=1e-3)


def test_number_variance_picket_fence():
    # A unit-spaced "picket fence" is maximally rigid: any integer-length window
    # holds exactly L levels (variance 0); a half-integer window holds L or L+1
    # with equal probability over the offset (variance 1/4).
    x = np.arange(20_000, dtype=np.float64)
    np.testing.assert_allclose(
        spacing.number_variance(x, [10.0, 25.0], n_offsets=3000), 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        spacing.number_variance(x, [10.5, 25.5], n_offsets=3000), 0.25, atol=1e-2
    )


def test_rigidity_poisson_statistics():
    # Uncorrelated (Poisson) levels: Sigma^2(L) ~ L and Delta_3(L) ~ L/15.
    rng = np.random.default_rng(0)
    n = 200_000
    x = np.sort(rng.uniform(0.0, n, size=n))
    lengths = np.array([5.0, 10.0, 20.0])
    sigma2 = spacing.number_variance(x, lengths, n_offsets=8000)
    delta3 = spacing.dyson_mehta_delta3(x, lengths, n_offsets=8000)
    np.testing.assert_allclose(sigma2, lengths, rtol=0.08)
    np.testing.assert_allclose(delta3, lengths / 15.0, rtol=0.08)


def test_number_variance_length_exceeding_span_is_nan():
    x = np.sort(np.random.default_rng(1).uniform(0.0, 50.0, size=200))
    out = spacing.number_variance(x, [10.0, 1000.0])
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_gpu_number_variance_matches_cpu():
    cp = pytest.importorskip("cupy")  # noqa: F841
    from zeta_spectral_gpu import spacing_gpu

    x = _sorted_levels(800)
    lengths = np.array([2.0, 5.0, 10.0])
    cpu = spacing.number_variance(x, lengths, n_offsets=500)
    gpu = spacing_gpu.number_variance_gpu(x, lengths, n_offsets=500)
    np.testing.assert_allclose(cpu, gpu, rtol=0, atol=1e-9)


def test_gpu_delta3_matches_cpu():
    cp = pytest.importorskip("cupy")  # noqa: F841
    from zeta_spectral_gpu import spacing_gpu

    x = _sorted_levels(800)
    lengths = np.array([2.0, 5.0, 10.0])
    cpu = spacing.dyson_mehta_delta3(x, lengths, n_offsets=500)
    gpu = spacing_gpu.dyson_mehta_delta3_gpu(x, lengths, n_offsets=500)
    np.testing.assert_allclose(cpu, gpu, rtol=1e-6, atol=1e-6)
