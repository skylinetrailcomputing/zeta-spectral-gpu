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
