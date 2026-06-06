"""Katz-Sarnak family statistics for quadratic Dirichlet L-functions (issue #51).

Fast tests pin the new number-theoretic input (the Kronecker-symbol quadratic
character, fundamental-discriminant enumeration) and the random-matrix kernels /
discrimination logic on synthetic data -- no mpmath. The ``slow`` tests run the
forward result: the family-averaged 1-level density of the quadratic Dirichlet
L-functions tracks the **symplectic** kernel and is suppressed at the central
point, with every member's zeros produced independently (never fit).
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import dirichlet as dl
from zeta_spectral_gpu import katz_sarnak as ks


# --- Quadratic character: the forward input (fast, pure number theory) --------


def test_kronecker_symbol_known_values():
    # Supplement (a | 2): +1 for a == +-1 (mod 8), -1 for a == +-3 (mod 8).
    assert ks.kronecker_symbol(7, 2) == 1  # 7 == -1 (mod 8)
    assert ks.kronecker_symbol(3, 2) == -1  # 3 (mod 8)
    assert ks.kronecker_symbol(2, 2) == 0  # even top
    # Legendre agreement for odd prime bottom: QRs mod 7 are {1, 2, 4}.
    qr7 = {1, 2, 4}
    for a in range(1, 7):
        assert ks.kronecker_symbol(a, 7) == (1 if a in qr7 else -1)
    # Edge cases (a | 0) and the sign factor (a | -1).
    assert ks.kronecker_symbol(1, 0) == 1 and ks.kronecker_symbol(2, 0) == 0
    assert ks.kronecker_symbol(-5, -1) == -1 and ks.kronecker_symbol(5, -1) == 1


def test_fundamental_discriminant_detection():
    # d == 1 (mod 4) squarefree, and d == 0 (mod 4) with d/4 == 2,3 (mod 4) squarefree.
    assert ks.is_fundamental_discriminant(5)  # 1 mod 4, squarefree
    assert ks.is_fundamental_discriminant(-3)  # 1 mod 4
    assert ks.is_fundamental_discriminant(-4)  # -4/4 = -1 == 3 mod 4
    assert ks.is_fundamental_discriminant(8) and ks.is_fundamental_discriminant(-8)
    assert ks.is_fundamental_discriminant(12)  # 12/4 = 3 == 3 mod 4, squarefree
    assert not ks.is_fundamental_discriminant(1)  # excluded (the zeta member)
    assert not ks.is_fundamental_discriminant(9)  # 1 mod 4 but 9 = 3^2 not squarefree
    assert not ks.is_fundamental_discriminant(20)  # 20/4 = 5 == 1 mod 4 -> not


def test_fundamental_discriminants_enumeration():
    fds = ks.fundamental_discriminants(13)
    assert fds == [-3, -4, 5, -7, -8, 8, -11, 12, 13]
    assert all(ks.is_fundamental_discriminant(d) for d in fds)


def test_quadratic_character_matches_existing_characters():
    # d = 5 (real, even): chi_5 = (5 | .) = Legendre (. | 5) = dirichlet(5, index 2).
    np.testing.assert_allclose(ks.quadratic_character(5), dl.dirichlet_character(5, 2))
    # d = -4 (imaginary, odd): chi_-4 = the Dirichlet beta character mod 4.
    np.testing.assert_allclose(ks.quadratic_character(-4), dl.dirichlet_character(4, 1))
    # Real primitive character: every value is real and chi^2 is principal.
    chi = ks.quadratic_character(-7)
    assert dl.is_real_character(chi)
    units = chi != 0
    np.testing.assert_allclose((chi[units] ** 2).real, 1.0)


def test_quadratic_character_rejects_nonfundamental():
    with pytest.raises(ValueError):
        ks.quadratic_character(12 + 8)  # 20 is not fundamental
    with pytest.raises(ValueError):
        ks.quadratic_character(1)


# --- Random-matrix kernels + discrimination (fast, no zeros) ------------------


def test_symmetry_kernel_values():
    # Central point: Sp suppressed to 0, U flat at 1, SO+ enhanced to 2.
    assert ks.symplectic_density(0.0) == pytest.approx(0.0)
    assert ks.unitary_density(0.0) == pytest.approx(1.0)
    assert ks.orthogonal_even_density(0.0) == pytest.approx(2.0)
    # All three decay to 1 away from the centre (sinc -> 0).
    far = np.array([10.25, 20.75])
    for kernel in (ks.symplectic_density, ks.orthogonal_even_density):
        np.testing.assert_allclose(kernel(far), 1.0, atol=0.05)


def test_one_level_density_normalisation():
    # Unit density: one zero per unit-x per member -> empirical W == 1 in every bin
    # (W = counts / (n_members * bin_width), bin_width == 1 here).
    n_members, x_max, n_bins = 10, 6.0, 6
    bin_centres = np.arange(n_bins) + 0.5  # width-1 bins on [0, 6)
    rescaled = np.repeat(bin_centres, n_members)  # n_members zeros per bin
    centres, w = ks.one_level_density(rescaled, n_members, x_max=x_max, n_bins=n_bins)
    np.testing.assert_allclose(centres, bin_centres)
    np.testing.assert_allclose(w, 1.0)


def test_classify_symmetry_on_kernel_samples():
    # Feeding each kernel's own values to the classifier returns that kernel.
    centres = np.linspace(0.05, 3.0, 24)
    assert ks.classify_symmetry(centres, ks.symplectic_density(centres)) == "symplectic"
    assert ks.classify_symmetry(centres, ks.unitary_density(centres)) == "unitary"
    assert (
        ks.classify_symmetry(centres, ks.orthogonal_even_density(centres))
        == "orthogonal_even"
    )


def test_conductor_density_positive_and_rescale():
    # log(q)/(2 pi) is positive for every fundamental discriminant (q >= 3).
    for d in ks.fundamental_discriminants(40):
        assert ks.conductor_density(abs(d)) > 0.0
    np.testing.assert_allclose(
        ks.rescale_zeros(np.array([1.0, 2.0]), 100),
        np.array([1.0, 2.0]) * np.log(100) / (2 * np.pi),
    )


# --- Forward producer: the locator at family scale (CPU path, no mpmath) ------


def test_locate_member_zeros_cpu_finds_beta_zeros():
    # FORWARD: the chi_-4 * mu partial sum |M'_z| peaks at the Dirichlet beta zeros,
    # located from the character alone (no zero consumed). CPU path; the GPU path is
    # exercised under importorskip below.
    grid = np.arange(2.0, 16.0, 0.01)
    peaks, backend = ks.locate_member_zeros(-4, 4000, grid, prefer_gpu=False)
    assert backend == "cpu"
    for known in (6.020949, 10.243770, 12.988056):
        assert peaks.size and np.min(np.abs(peaks - known)) < 0.05


# --- Forward result: the quadratic family is symplectic (slow, mpmath) --------


@pytest.mark.slow
def test_quadratic_character_zeros_match_beta():
    # The KS module's character feeds the independent mpmath zero finder correctly.
    zeros = dl.lfunction_zeros(ks.quadratic_character(-4), e_max=14.0)
    for known in (6.020949, 10.243770, 12.988056):
        assert np.min(np.abs(zeros - known)) < 1e-3


@pytest.mark.slow
def test_quadratic_family_one_level_density_is_symplectic():
    # THE forward result: pool the independently-computed low-lying zeros of the
    # quadratic Dirichlet family, rescale by conductor, and the empirical 1-level
    # density tracks the SYMPLECTIC kernel -- decisively suppressed near the central
    # point (unitary is flat at 1, even-orthogonal enhanced to ~2 there). Zeros are
    # produced per member and only compared; nothing is fit.
    family = ks.fundamental_discriminants(48)
    assert len(family) >= 25
    centres, w, n_members = ks.family_one_level_density(
        family, x_max=3.0, n_bins=12, dps=12
    )
    # Verdict: symplectic is the closest kernel near the centre.
    distances = ks.symmetry_distances(centres, w, x_disc=1.0)
    assert min(distances, key=distances.get) == "symplectic"
    assert distances["symplectic"] < distances["unitary"]
    assert distances["unitary"] < distances["orthogonal_even"]
    # Central suppression: the lowest bins sit far below the unitary baseline of 1.
    central = w[centres < 0.5]
    assert central.mean() < 0.6


@pytest.mark.slow
def test_locator_matches_mpmath_low_zeros_real_character():
    # Cross-check the forward producer against the ground truth on a family member:
    # the locator's low peaks coincide with the independent mpmath zeros.
    d = -7
    grid = np.arange(0.5, 18.0, 0.01)
    peaks, _ = ks.locate_member_zeros(d, 6000, grid, prefer_gpu=False)
    true = dl.lfunction_zeros(ks.quadratic_character(d), e_min=0.5, e_max=18.0)
    assert true.size >= 4
    matched = sum(bool(peaks.size) and np.min(np.abs(peaks - t)) < 0.2 for t in true)
    assert matched >= true.size - 1


# --- GPU parity for the forward producer (importorskip) -----------------------


@pytest.mark.slow
def test_locate_member_zeros_gpu_matches_cpu():
    pytest.importorskip("cupy")
    grid = np.arange(2.0, 16.0, 0.01)
    gpu_peaks, backend = ks.locate_member_zeros(-4, 4000, grid, prefer_gpu=True)
    assert backend == "gpu"
    cpu_peaks, _ = ks.locate_member_zeros(-4, 4000, grid, prefer_gpu=False)
    # Same located ordinates from both backends (fp64 locator).
    assert gpu_peaks.size == cpu_peaks.size
    np.testing.assert_allclose(np.sort(gpu_peaks), np.sort(cpu_peaks), atol=2e-2)
