"""Forward L-function generalisation of the prime-driven mirror locator (#25).

Sierra's mirror model goes from zeta to any Dirichlet ``L``-function by switching
the reflection coefficients ``mu(n)/sqrt(n) -> chi(n) mu(n)/sqrt(n)`` (eq. 13.6).
The decisive forward checks (``test_locator_peaks_at_lfunction_zeros_*``) are that
the ``chi*mu`` partial sum ``|M'_z|`` peaks at the *independently computed* zeros
of ``L(s, chi)`` — the character depends only on its modulus, so no zero is fed
in. The fast tests pin the new number-theoretic input (orthogonality,
multiplicativity, the Dirichlet beta cross-check); the ``slow`` tests run the
mpmath ground-truth zero scans.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu import dirichlet as dl


def _peak_ordinates(grid: np.ndarray, abs_m: np.ndarray, height: float) -> np.ndarray:
    interior = (
        (abs_m[1:-1] > abs_m[:-2]) & (abs_m[1:-1] >= abs_m[2:]) & (abs_m[1:-1] > height)
    )
    return grid[np.nonzero(interior)[0] + 1]


# --- Character algebra: the forward input (fast, pure number theory) ----------


def test_primitive_root_mod_7():
    assert dl.primitive_root(7) == 3  # smallest generator of (Z/7Z)*
    assert sorted(pow(3, m, 7) for m in range(6)) == [1, 2, 3, 4, 5, 6]


def test_principal_character_is_indicator_of_units():
    chi = dl.dirichlet_character(7, 0)
    assert chi[0] == 0
    assert np.allclose(chi[1:], 1.0)  # 1 on every unit mod 7


def test_quadratic_character_matches_legendre_symbol():
    # index (p-1)/2 is the real quadratic character = the Legendre symbol.
    p = 7
    chi = dl.dirichlet_character(p, (p - 1) // 2)
    assert dl.is_real_character(chi)
    qr = {pow(a, 2, p) for a in range(1, p)}
    for a in range(1, p):
        expected = 1.0 if a in qr else -1.0
        assert np.isclose(chi[a].real, expected) and np.isclose(chi[a].imag, 0.0)


def test_character_orthogonality_and_multiplicativity():
    p = 5
    chars = [dl.dirichlet_character(p, j) for j in range(p - 1)]
    # Orthogonality of characters: sum_a chi_i(a) conj(chi_j(a)) = (p-1) delta_ij.
    for i in range(p - 1):
        for j in range(p - 1):
            s = sum(chars[i][a] * np.conj(chars[j][a]) for a in range(p))
            assert np.isclose(s, (p - 1) if i == j else 0.0, atol=1e-10)
    # Complete multiplicativity of a genuinely complex character.
    chi = chars[1]
    assert not dl.is_real_character(chi)
    for a in range(1, p):
        for b in range(1, p):
            assert np.isclose(chi[(a * b) % p], chi[a] * chi[b], atol=1e-12)


def test_mod4_odd_character_is_dirichlet_beta():
    # L(2, chi_4) = Catalan's constant pins the character to Dirichlet beta.
    import mpmath as mp

    chi4 = dl.dirichlet_character(4, 1)
    assert list(np.real(chi4)) == [0, 1, 0, -1]
    assert abs(complex(dl.lfunction_value(2, chi4)) - float(mp.catalan)) < 1e-12


def test_composite_modulus_rejected():
    with pytest.raises(NotImplementedError):
        dl.dirichlet_character(15, 1)  # only prime moduli (and 4) supported


def test_lfunction_weights_dtype_and_trivial_character():
    n = 200
    mu = dm.mobius_sieve(n)
    # Trivial (period-1, all-ones) character -> weights are exactly mu (real path).
    trivial = np.ones(1, dtype=np.complex128)
    w = dl.lfunction_weights(trivial, n, mu=mu)
    assert w.dtype == np.float64
    np.testing.assert_array_equal(w, mu[1 : n + 1].astype(np.float64))
    # Genuinely complex character -> complex weights (the weighted_locator path).
    wc = dl.lfunction_weights(dl.dirichlet_character(5, 1), n)
    assert wc.dtype == np.complex128


# --- Ground-truth zeros + forward acceptance (slow: mpmath scans) -------------


@pytest.mark.slow
def test_lfunction_zeros_dirichlet_beta():
    # Independently-computed zeros of L(s, chi_4) match the known beta ordinates.
    zeros = dl.lfunction_zeros(dl.dirichlet_character(4, 1), e_max=14.0)
    for known in (6.020949, 10.243770, 12.988056):
        assert np.min(np.abs(zeros - known)) < 1e-3


@pytest.mark.slow
def test_principal_character_recovers_zeta_zeros():
    # L(s, chi_0 mod 3) = zeta(s)(1 - 3^-s): same zeros as zeta on the line.
    zeros = dl.lfunction_zeros(dl.dirichlet_character(3, 0), e_max=22.0)
    for gamma in (14.134725, 21.022040):
        assert np.min(np.abs(zeros - gamma)) < 1e-3


@pytest.mark.slow
def test_locator_peaks_at_lfunction_zeros_real_character():
    # FORWARD: the chi*mu partial sum |M'_z| peaks at the zeros of the Dirichlet
    # beta L-function, fed only chi (mod 4) and mu -- no zero consumed.
    chi4 = dl.dirichlet_character(4, 1)
    true = dl.lfunction_zeros(chi4, e_max=20.0)
    assert true.size >= 4
    n = 4000
    w = dl.lfunction_weights(chi4, n)
    assert w.dtype == np.float64  # real character keeps the fast real locator
    grid = np.arange(2.0, 20.0, 0.01)
    abs_m = np.abs(dm.mobius_partial_sum(grid, n, weights=w))
    peaks = _peak_ordinates(grid, abs_m, 0.3 * np.log(n))
    matched = sum(bool(peaks.size) and np.min(np.abs(peaks - t)) < 0.2 for t in true)
    assert matched >= true.size - 1  # locate essentially every L-zero


@pytest.mark.slow
def test_locator_peaks_at_lfunction_zeros_complex_character():
    # FORWARD with a genuinely COMPLEX character (mod 5): complex chi*mu weights,
    # |M'_z| still peaks at the independently-computed zeros, which are *asymmetric*
    # in E (a different L-function, not zeta).
    chi5 = dl.dirichlet_character(5, 1)
    assert not dl.is_real_character(chi5)
    true = dl.lfunction_zeros(chi5, e_min=-5.0, e_max=15.0)
    assert true.size >= 4
    n = 5000
    w = dl.lfunction_weights(chi5, n)
    assert w.dtype == np.complex128
    grid = np.arange(-5.0, 15.0, 0.01)
    abs_m = np.abs(dm.mobius_partial_sum(grid, n, weights=w))
    peaks = _peak_ordinates(grid, abs_m, 0.3 * np.log(n))
    matched = sum(bool(peaks.size) and np.min(np.abs(peaks - t)) < 0.25 for t in true)
    assert matched >= true.size - 1
