"""GPU locator invariants for Sierra's prime-driven Moebius-mirror model (#25, Phase 2).

The CPU forward locator :func:`dirac_mirror.mobius_partial_sum` is the truth; the
kernel-backed scan in :mod:`dirac_mirror_gpu` must reproduce it to floating-point
tolerance (the house GPU-vs-CPU rule) and -- fed only ``mu(k)`` and ``k^{-z}`` --
still locate the Riemann zeros as peaks of ``|M'_z|`` on an ``E``-grid (Sierra's
Fig. 14). GPU tests skip cleanly when cupy isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu import dirac_mirror_gpu as gpu
from zeta_spectral_gpu.zeros import load_ordinates

ZERO_1 = 14.134725  # probe points only -- never operator input
ZERO_2 = 21.022040


def _local_maxima(y: np.ndarray, height: float) -> np.ndarray:
    interior = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > height)
    return np.nonzero(interior)[0] + 1


def test_gpu_matches_cpu_locator():
    # Decisive correctness check: the kernel scan reproduces the numpy
    # outer-product reference across the grid, for small and larger truncations.
    pytest.importorskip("cupy")
    grid = np.arange(10.0, 30.0, 0.05)
    for n in (200, 1500):
        g = gpu.mobius_partial_sum_gpu(grid, n)
        c = dm.mobius_partial_sum(grid, n)
        np.testing.assert_allclose(g, c, rtol=0, atol=1e-9)


def test_gpu_scalar_matches_cpu():
    pytest.importorskip("cupy")
    g = gpu.mobius_partial_sum_gpu(ZERO_1, 300)
    c = complex(dm.mobius_partial_sum(ZERO_1, 300))
    assert isinstance(g, complex)
    assert abs(g - c) < 1e-9


def test_gpu_preserves_grid_shape():
    pytest.importorskip("cupy")
    grid = np.linspace(12.0, 22.0, 37)
    out = gpu.mobius_partial_sum_gpu(grid, 250)
    assert out.shape == grid.shape
    assert out.dtype == np.complex128


def test_gpu_locator_finds_zeros_forward():
    # Forward: |M'_z| peaks land on the true ordinates, fed only mu(k) and k^{-z}.
    # The zeros are used solely to score the located peaks (a downstream check).
    pytest.importorskip("cupy")
    grid = np.arange(10.0, 40.0, 0.01)
    n = 3000
    absM = np.abs(gpu.mobius_partial_sum_gpu(grid, n))
    peaks = grid[_local_maxima(absM, 0.5 * np.log(n))]
    true = load_ordinates(20, source="mpmath")
    true = true[true < 40.0]
    matched = sum(
        bool(peaks.size) and float(np.min(np.abs(peaks - t))) < 0.3 for t in true
    )
    assert matched >= true.size - 1  # locate essentially every zero below E=40


def test_gpu_lfunction_weights_hook():
    # Dirichlet-L generalisation (eq. 13.6): swap mu(k) -> chi(k) mu(k). Trivial
    # character recovers the zeta weights bit-for-bit through the GPU path.
    pytest.importorskip("cupy")
    n = 200
    mu = dm.mobius_sieve(n)
    chi = np.ones(n + 1)
    weights = chi[1 : n + 1] * mu[1 : n + 1]
    a = gpu.mobius_partial_sum_gpu(ZERO_1, n, weights=weights)
    b = gpu.mobius_partial_sum_gpu(ZERO_1, n, mu=mu)
    assert abs(a - b) < 1e-12


def test_gpu_complex_character_matches_cpu():
    # Genuinely complex Dirichlet character (mod 5) -> complex chi*mu weights ->
    # the weighted_locator kernel. Must reproduce the numpy reference across the
    # grid (the house GPU-vs-CPU rule), including the asymmetric negative-E zeros.
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import dirichlet as dl

    chi5 = dl.dirichlet_character(5, 1)
    n = 1500
    w = dl.lfunction_weights(chi5, n)
    assert w.dtype == np.complex128
    grid = np.arange(-5.0, 18.0, 0.05)
    g = gpu.mobius_partial_sum_gpu(grid, n, weights=w)
    c = dm.mobius_partial_sum(grid, n, weights=w)
    np.testing.assert_allclose(g, c, rtol=0, atol=1e-9)
