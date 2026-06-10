"""GPU zero Fourier statistic vs the CPU reference (issue #84, GPU half).

The CPU ``arithmetic_correlations.zero_fourier`` is the truth; the kernel in
``kernels/arithmetic_correlations.cu`` must reproduce it to floating-point
tolerance (the house GPU-vs-CPU rule). Skips cleanly when cupy isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import arithmetic_correlations as ac


def _synthetic_levels(n=5000, lo=1000.0, hi=3000.0, seed=84):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, n))


@pytest.mark.parametrize("kind", ["rect", "hann"])
def test_gpu_zero_fourier_matches_cpu(kind):
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

    tau = _synthetic_levels()
    u = np.linspace(0.1, 4.0, 257)
    cpu = ac.zero_fourier(tau, u, 1200.0, 2800.0, window=kind)
    gpu = acg.zero_fourier_gpu(tau, u, 1200.0, 2800.0, window=kind)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-8)


def test_gpu_zero_fourier_scalar_frequency():
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

    tau = _synthetic_levels(n=500)
    cpu = ac.zero_fourier(tau, 0.7, 1200.0, 2800.0)
    gpu = acg.zero_fourier_gpu(tau, 0.7, 1200.0, 2800.0)
    assert gpu.shape == cpu.shape == (1,)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)


def test_gpu_zero_fourier_large_ordinates():
    """Heights ~1e5 stress the sincos argument reduction; the kernel must still
    track numpy bit-for-bit at tolerance."""
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

    tau = _synthetic_levels(n=20_000, lo=3.0e4, hi=7.0e4)
    u = np.linspace(0.5, 4.0, 101)
    cpu = ac.zero_fourier(tau, u, 3.2e4, 6.8e4)
    gpu = acg.zero_fourier_gpu(tau, u, 3.2e4, 6.8e4)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-7)
