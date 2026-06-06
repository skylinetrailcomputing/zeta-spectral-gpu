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
