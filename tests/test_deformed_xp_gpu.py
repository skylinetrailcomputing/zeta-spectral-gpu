"""GPU dense-eigensolve invariants for Sierra's deformed-xp operator (#31).

The CPU secular reference ``deformed_xp.secular_spectrum`` (real roots of the
Bessel-K equation, eq. 14) is the truth spectrum; the Galerkin assembly + eigh
here must reproduce it for the resolved low modes (the house GPU-vs-CPU rule),
with no spurious particle-in-a-box modes. GPU tests skip cleanly without cupy so
the CPU invariants still run on a bare environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import deformed_xp_gpu as gpu


def test_reduced_matrix_is_hermitian():
    # The eq. 11 constraint projection makes the reduced matrix exactly Hermitian
    # (the rank-2 boundary defect vanishes on the constraint null space).
    h = gpu.galerkin_matrix(40)
    assert h.shape == (39, 39)
    assert np.max(np.abs(h - h.conj().T)) < 1e-12


def test_spectrum_matches_secular_reference():
    # Decisive correctness check: the assembled spectrum reproduces the secular
    # roots (#23) to floating-point precision for the resolved low modes.
    ref = dxp.secular_spectrum(8, cache_dir=None)
    spec = gpu.spectrum(80, use_gpu=False)
    assert spec.size >= 8
    np.testing.assert_allclose(spec[:8], ref, atol=1e-8)


def test_no_modes_below_classical_floor():
    # The decaying Galerkin basis removes the spurious box spectrum: every level
    # sits above the classical floor |E| >= 2h = 4*pi (eq. 6).
    spec = gpu.spectrum(60, use_gpu=False)
    assert spec.size > 0
    assert np.all(spec > dxp.CLASSICAL_BOUND - 1e-6)


def test_convergence_improves_with_basis_size():
    # Spectral convergence to the secular roots as the basis grows.
    ref = dxp.secular_spectrum(5, cache_dir=None)
    err_small = np.abs(gpu.spectrum(30, use_gpu=False)[:5] - ref).max()
    err_large = np.abs(gpu.spectrum(70, use_gpu=False)[:5] - ref).max()
    assert err_large < err_small
    assert err_large < 1e-9


def test_theta_changes_the_spectrum():
    # theta parameterises the U(1) self-adjoint extensions; a different angle is a
    # genuinely different operator -> a shifted spectrum (the extension is wired in).
    a = gpu.spectrum(50, theta=np.pi / 4, use_gpu=False)
    b = gpu.spectrum(50, theta=np.pi / 2, use_gpu=False)
    assert np.max(np.abs(a[:5] - b[:5])) > 1e-3


def test_gpu_matches_cpu_eigh():
    pytest.importorskip("cupy")
    h = gpu.galerkin_matrix(60)
    cpu = gpu.eigenvalues(h, use_gpu=False)
    gpu_eig = gpu.eigenvalues(h, use_gpu=True)
    np.testing.assert_allclose(gpu_eig, cpu, rtol=0, atol=1e-9)


def test_gpu_spectrum_matches_secular_reference():
    pytest.importorskip("cupy")
    ref = dxp.secular_spectrum(8, cache_dir=None)
    spec = gpu.spectrum(80, use_gpu=True)
    np.testing.assert_allclose(spec[:8], ref, atol=1e-8)
