"""Exact finite-eps transfer-matrix machinery for Sierra's models (#44 spike, #45).

These tests back the GO/NO-GO ruling in ``_private/issue-44-resonance-ruling.md``.
Two threads:

* **Harmonic model (Appendix A) — the no-prime positive control.** The one model
  in the paper with a genuine *finite-eps* spectrum: exact continuum bands
  (eq. A16) and discrete levels ``E_n = 2 pi n`` (eq. A13). The exact-product
  code reproduces both. Its periods are *integers*, not ``log p`` — so the place
  a finite-eps density exists is the place with no primes.
* **Moebius (Riemann) model — the negative result.** At finite eps the exact
  product equals the eps -> 0 BCH locator up to vanishing higher-order
  corrections, and carries **no** operator-intrinsic discrete spectrum: a bound
  state appears only with the per-zero ``vartheta`` tuning (Piece B, inverse).
"""

from __future__ import annotations

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm

ZERO_1 = 14.134725
NONZERO = 17.0
SIGMA_Z = np.diag([1.0, -1.0]).astype(np.complex128)


# --- the exact single-mirror transfer matrix (eq. 115) ------------------------


def test_transfer_matrix_is_su11():
    # det T = 1 and T^dag sigma_z T = sigma_z (pseudo-unitary), real and complex p.
    for varrho in (0.3, -0.2, 0.15 + 0.07j):
        T = dm.transfer_matrix(11.5, varrho, np.sqrt(7.0))
        assert abs(np.linalg.det(T) - 1.0) < 1e-13
        assert np.max(np.abs(T.conj().T @ SIGMA_Z @ T - SIGMA_Z)) < 1e-13


def test_amplitude_norms_matches_brute_force_product():
    # The vectorised recursion equals an explicit T_n^{-1} ... T_2^{-1} product.
    E, eps, n, vt = NONZERO, 0.12, 60, 0.7
    mu = dm.mobius_sieve(n)
    A = np.array([1.0, np.exp(1j * vt)], dtype=np.complex128)  # |A_1(vartheta)>
    brute = [abs(A[0]) ** 2 + abs(A[1]) ** 2]
    for k in range(2, n + 1):
        v = eps * mu[k] / np.sqrt(k)
        if v != 0:
            A = np.linalg.inv(dm.transfer_matrix(E, v, np.sqrt(k))) @ A
        brute.append(abs(A[0]) ** 2 + abs(A[1]) ** 2)
    got = dm.mobius_amplitude_norms(E, n, eps=eps, vartheta=vt)[0]
    np.testing.assert_allclose(got, brute, atol=1e-11)


# --- finite eps adds nothing: exact product -> BCH locator as eps -> 0 ---------


def test_exact_product_collapses_to_bch_locator():
    # max|exact - BCH| vanishes at least as fast as eps^2 (empirically eps^3):
    # finite eps adds only vanishing higher-order corrections to the eps -> 0
    # locator, no new spectral structure. Checked at a zero and off a zero.
    for E in (ZERO_1, NONZERO):
        errs = []
        for eps in (0.08, 0.04, 0.02):
            ex = dm.mobius_amplitude_norms(E, 120, eps=eps, vartheta=0.3)[0]
            bch = dm.mobius_amplitude_norms(E, 120, eps=eps, vartheta=0.3, bch=True)[0]
            errs.append(np.max(np.abs(ex - bch)[2:]))
        # halving eps shrinks the gap by >= ~3.5x (O(eps^2) or better)
        assert errs[0] / errs[1] > 3.5
        assert errs[1] / errs[2] > 3.5
        assert errs[-1] < 1e-3  # tiny in absolute terms at eps = 0.02


def test_mobius_finite_eps_has_no_intrinsic_bound_state():
    # The decisive negative result. At finite eps the Moebius spectrum is a
    # continuum: at a zero with a *generic* (structural) vartheta the exact norm
    # does NOT decay -- no bound state. A normalizable state appears only when
    # vartheta is tuned to the zero (Piece B, the inverse trap), reproducing
    # Sierra's Fig. 4 with the exact (un-BCH) product.
    tr = np.arange(1, 4001)
    generic = dm.mobius_amplitude_norms(ZERO_1, 4000, eps=0.1, vartheta=0.0)[0]
    tuned_vt = dm.riemann_tuning_phase(ZERO_1)  # uses theta(E) => Piece B
    tuned = dm.mobius_amplitude_norms(ZERO_1, 4000, eps=0.1, vartheta=tuned_vt)[0]
    assert generic[-500:].mean() > 0.5  # generic vartheta: no decay (continuum)
    assert tuned[-500:].mean() < 0.1  # tuned vartheta: decays (bound state)
    assert tuned[-1] < tuned[0]
    assert len(tr) == generic.size


# --- harmonic model (Appendix A): the no-prime finite-eps positive control -----


def test_harmonic_band_formula_matches_eq_A16():
    eps = 0.2
    delta = dm.harmonic_gap_half_width(eps)
    assert abs(np.sin(np.pi * delta) - 2 * eps / (1 + eps**2)) < 1e-13
    # scanned continuum region matches the analytic band edges (eq. A16).
    Es = np.linspace(2 * np.pi, 4 * np.pi, 40001)
    cont = dm.harmonic_is_continuum(Es, eps)
    lo, hi = Es[np.argmax(cont)], Es[len(cont) - 1 - np.argmax(cont[::-1])]
    band_lo, band_hi = dm.harmonic_bands(eps, [1])[0]
    assert abs(lo - band_lo) < 1e-2 and abs(hi - band_hi) < 1e-2


def test_harmonic_step_matrix_trace_and_det():
    eps, E = 0.2, 3.3
    S = dm.harmonic_step_matrix(E, eps)
    assert abs(np.linalg.det(S) - 1.0) < 1e-13
    assert abs(np.trace(S) - dm.harmonic_trace(E, eps)) < 1e-13
    assert abs(dm.harmonic_trace(E, eps).imag) < 1e-13


def test_harmonic_exact_product_band_bounded_gap_grows():
    # Same exact-product code as the Moebius model, switched to integer periods:
    # bounded norm in a band (continuum), exponential growth in a gap.
    eps = 0.2
    e_band = 2 * np.pi * 1.5  # middle of band m = 1
    e_gap = 2 * np.pi * 2 - 0.1  # inside the gap around 4 pi
    norms = dm.harmonic_amplitude_norms(np.array([e_band, e_gap]), 200, eps=eps)
    assert norms[0].max() < 10.0  # band: bounded
    assert norms[1, -1] > 1e30  # gap: blows up


def test_harmonic_discrete_level_is_a_bound_state():
    # E_n = 2 pi n with vartheta = 0 is a genuine normalizable eigenstate
    # (eq. A13): the exact norm decays exponentially. Driven by the integers,
    # not the primes -- the contrast that makes the Moebius NO-GO sharp.
    norms = dm.harmonic_amplitude_norms(2 * np.pi, 40, eps=0.2, vartheta=0.0)[0]
    assert norms[0] == 2.0
    assert norms[20] < 1e-5
    assert norms[39] < norms[20]  # monotone decay
