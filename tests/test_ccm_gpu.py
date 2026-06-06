"""Invariants for the CCM GPU / fp64 paths (#9).

The decisive tests mirror the repo's "CPU reference first" rule: the fp64 numpy
fill must reproduce the mpmath reference (:mod:`ccm`), and the hand-written CUDA
kernel must reproduce that fp64 fill. The conditioning test pins the *precision
wall* — fp64 ``eigh`` recovers the minimal eigenvalue only while it stays above
the fp64 floor, and saturates beyond it.

GPU tests skip cleanly when cupy isn't installed; the fp64/mpmath invariants run
on a bare environment.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from zeta_spectral_gpu import ccm, ccm_gpu


def _mpmath_matrix_to_fp64(A, dim: int) -> np.ndarray:
    return np.array([[float(A[i, j]) for j in range(dim)] for i in range(dim)])


def test_digamma_trigamma_fp64_match_mpmath():
    mp.mp.dps = 30
    z = np.array(
        [0.25 + 0j, 0.25 + 0.7j, 0.25 + 3.1j, 0.25 + 40j, 1.0 + 0j], dtype=complex
    )
    dg = ccm_gpu._digamma_c(z)
    tg = ccm_gpu._trigamma_c(z)
    for zi, d, t in zip(z, dg, tg):
        assert abs(d - complex(mp.digamma(complex(zi)))) < 1e-13
        assert abs(t - complex(mp.polygamma(1, complex(zi)))) < 1e-13


def test_lerch_fp64_matches_mpmath():
    # Direct geometric sum equals mpmath's Lerch for the small-|q| regime we use.
    mp.mp.dps = 40
    L = 2 * float(mp.log(mp.sqrt(13)))
    q = float(mp.exp(-2 * L))
    a = np.array([0.25 + 0j, 0.25 + 1j * np.pi * 3 / L], dtype=complex)
    for s in (1, 2):
        got = ccm_gpu._lerch_fp64(q, s, a)
        for g, av in zip(got, a):
            assert abs(g - complex(mp.lerchphi(q, s, complex(av)))) < 1e-13


def test_fp64_fill_matches_mpmath_reference():
    # The fp64 numpy assembly reproduces the mpmath fill to near machine epsilon.
    mp.mp.dps = 50
    for N, x in [(8, 13), (16, 14)]:
        dim = 2 * N + 1
        Aref = _mpmath_matrix_to_fp64(ccm.assemble_weil_matrix(N, mp.sqrt(x)), dim)
        Afp = ccm_gpu.assemble_weil_matrix_fp64(N, float(np.sqrt(x)))
        scale = np.max(np.abs(Aref))
        assert np.max(np.abs(Afp - Aref)) < 1e-10 * scale


def test_fp64_fill_symmetric():
    Afp = ccm_gpu.assemble_weil_matrix_fp64(12, float(np.sqrt(13)))
    np.testing.assert_allclose(Afp, Afp.T, rtol=0, atol=0)


def test_operator_spectrum_matches_ordinates():
    # Forward: the prime-built operator's spectrum reproduces the low zeros, with
    # the zeros used only to *check* (the helper returns no zeros itself).
    spec = ccm.operator_spectrum(20, mp.sqrt(13), count=3, dps=80)
    zeros = ccm.reference_ordinates(3)
    assert len(spec) == 3
    assert all(e > 0 for e in spec)
    assert all(np.diff([float(s) for s in spec]) > 0)
    assert abs(spec[0] - zeros[0]) < mp.mpf(10) ** (-20)  # t_1 far below fp64


def test_gpu_assembly_matches_fp64_reference():
    cp = pytest.importorskip("cupy")  # noqa: F841
    for N, x in [(8, 13), (20, 14), (40, 9)]:
        ref = ccm_gpu.assemble_weil_matrix_fp64(N, float(np.sqrt(x)))
        gpu = cp.asnumpy(ccm_gpu.assemble_weil_matrix_gpu(N, float(np.sqrt(x))))
        np.testing.assert_allclose(gpu, ref, rtol=0, atol=1e-12)


def test_conditioning_fp64_precision_wall():
    cp = pytest.importorskip("cupy")  # noqa: F841
    pytest.importorskip("cupyx")
    try:
        cp.linalg.eigh(cp.eye(2, dtype=cp.float64))
    except Exception:  # pragma: no cover - cuSOLVER unavailable
        pytest.skip("cuSOLVER eigh unavailable")

    mp.mp.dps = 80
    N = 40
    # x = 4: eps_N sits above the fp64 floor, so fp64 eigh recovers it.
    A4 = ccm.assemble_weil_matrix(N, mp.sqrt(4))
    eps_mp4 = float(ccm.smallest_even_eigenvector(A4, N).eigenvalue)
    cond4 = ccm_gpu.conditioning_fp64(ccm_gpu.assemble_weil_matrix_gpu(N, 2.0))
    assert eps_mp4 > 1e-13
    assert abs(cond4.eps_n - eps_mp4) < 1e-2 * eps_mp4

    # x = 12: eps_N is far below the fp64 floor, so fp64 saturates (the wall).
    A12 = ccm.assemble_weil_matrix(N, mp.sqrt(12))
    eps_mp12 = float(ccm.smallest_even_eigenvector(A12, N).eigenvalue)
    cond12 = ccm_gpu.conditioning_fp64(
        ccm_gpu.assemble_weil_matrix_gpu(N, float(np.sqrt(12)))
    )
    assert eps_mp12 < 1e-25
    # fp64 cannot resolve the true near-null mode: its eps is orders too large and
    # its condition number saturates orders below the true ~eig_max/eps_N.
    assert cond12.eps_n > 1e5 * eps_mp12
    assert cond12.cond < 1e-5 * (cond12.eig_max / eps_mp12)


def test_ensure_cuda_libs_idempotent():
    # Bootstrap is safe to call repeatedly and on import; must not raise.
    ccm_gpu._ensure_cuda_libs()
    ccm_gpu._ensure_cuda_libs()
