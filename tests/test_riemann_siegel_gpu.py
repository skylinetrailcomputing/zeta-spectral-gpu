"""GPU Riemann-Siegel main sum vs the CPU reference (issue #55, GPU half).

The CPU evaluator :mod:`riemann_siegel` is the truth; the kernel-backed main sum
in :mod:`riemann_siegel_gpu` must reproduce it to floating-point tolerance (the
house GPU-vs-CPU rule) and -- fed only ``t`` -- still locate the on-line zeros as
sign changes of ``Z``. GPU tests skip cleanly when cupy isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import riemann_siegel as rs
from zeta_spectral_gpu import riemann_siegel_gpu as gpu


def test_gpu_main_sum_matches_cpu():
    # Decisive correctness check: the kernel reproduces the numpy main sum across a
    # wide height grid (the embarrassingly-parallel piece the GPU owns).
    pytest.importorskip("cupy")
    grid = np.arange(100.0, 5000.0, 0.25)
    g = gpu.main_sum_gpu(grid)
    c = rs.main_sum(grid)
    np.testing.assert_allclose(g, c, rtol=0, atol=1e-9)


def test_gpu_hardy_z_matches_cpu():
    pytest.importorskip("cupy")
    grid = np.linspace(80.0, 3000.0, 4000)
    np.testing.assert_allclose(
        gpu.hardy_z_gpu(grid), rs.hardy_z(grid), rtol=0, atol=1e-9
    )


def test_gpu_zeta_critical_matches_cpu():
    pytest.importorskip("cupy")
    grid = np.linspace(120.0, 2000.0, 1500)
    np.testing.assert_allclose(
        gpu.zeta_critical_gpu(grid), rs.zeta_critical(grid), rtol=0, atol=1e-9
    )


def test_gpu_scalar_and_high_point_match_cpu():
    pytest.importorskip("cupy")
    g = gpu.hardy_z_gpu(1000.0)
    assert isinstance(g, float)
    assert abs(g - rs.hardy_z(1000.0)) < 1e-9
    # A single very large height (many main-sum terms) still matches the host path.
    assert abs(gpu.hardy_z_gpu(1.0e6) - rs.hardy_z(1.0e6)) < 1e-8


def test_gpu_preserves_grid_shape():
    pytest.importorskip("cupy")
    grid = np.linspace(100.0, 300.0, 35).reshape(5, 7)
    out = gpu.hardy_z_gpu(grid)
    assert out.shape == grid.shape
    assert out.dtype == np.float64


def test_gpu_locates_zeros_forward():
    # Forward: |Z| sign changes from the GPU scan land on the same zeros the CPU
    # finds, fed only the heights.
    pytest.importorskip("cupy")
    grid = np.arange(100.0, 200.0, 0.01)
    zg = gpu.hardy_z_gpu(grid)
    zc = rs.hardy_z(grid)
    sg = np.signbit(zg)
    sc = np.signbit(zc)
    # Identical sign pattern => identical located zeros (and a healthy count).
    assert np.array_equal(sg, sc)
    assert int(np.count_nonzero(sg[:-1] != sg[1:])) > 40
