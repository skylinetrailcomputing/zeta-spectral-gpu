"""Invariants for the batched-family Dirichlet-L locator (issue #68).

Phase-2 of the Katz-Sarnak family work: the family is embarrassingly parallel, so
the batched kernel (:mod:`dirichlet_locator_family_gpu`,
``kernels/dirichlet_locator_family.cu``) scans the whole family's ``|M'_z(E)|`` in
one launch. The decisive checks: the **CPU reference** (:func:`family_scan`) agrees
with the established single-character locator; the packing reconstructs each
character; the batched GPU kernel reproduces the CPU reference to floating-point
tolerance (the house GPU-vs-CPU rule) and reduces, on a one-member family, to the
single-character kernel; and the forward guarantee holds (no zero is ever read).
GPU tests skip cleanly when cupy isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import dirichlet as dl
from zeta_spectral_gpu import dirichlet_locator as dloc
from zeta_spectral_gpu import dirichlet_locator_family as fam
from zeta_spectral_gpu import katz_sarnak as ks

FAMILY = [-3, -4, 5, -7, 8, -8, -11]  # fundamental discriminants (all real chars)


def _chars(discs: list[int]) -> list[np.ndarray]:
    return [ks.quadratic_character(d) for d in discs]


# --- CPU reference + packing (fast, no cupy) ----------------------------------


def test_family_scan_matches_single_character_locator():
    # The batched CPU reference row == the established single-character scan
    # (dirichlet_locator.scan_locator, CPU path) for every member.
    chars = _chars(FAMILY)
    n = 1200
    grid = np.arange(1.0, 16.0, 0.05)
    block = fam.family_scan(chars, n, grid)
    assert block.shape == (len(chars), grid.size)
    assert block.dtype == np.complex128
    for m, char in enumerate(chars):
        ref, backend = dloc.scan_locator(char, n, grid, prefer_gpu=False)
        assert backend == "cpu"
        np.testing.assert_allclose(block[m], ref, rtol=0, atol=1e-12)


def test_pack_family_layout_and_realness():
    # Real-only family -> real pack; offsets/periods index each character table.
    chars = _chars(FAMILY)
    packed = fam.pack_family(chars, 500)
    assert packed.is_real and packed.chi_im is None
    assert packed.num_members == len(chars)
    np.testing.assert_array_equal(packed.periods, [c.size for c in chars])
    for m, char in enumerate(chars):
        off, q = int(packed.offsets[m]), int(packed.periods[m])
        np.testing.assert_allclose(packed.chi_re[off : off + q], char.real)
    # A genuinely complex member upgrades the whole pack to the complex layout.
    mixed = chars + [dl.dirichlet_character(5, 1)]
    pm = fam.pack_family(mixed, 500)
    assert not pm.is_real and pm.chi_im is not None
    assert pm.chi_im.shape == pm.chi_re.shape


def test_pack_family_shared_arrays():
    # mu_amp[k-1] = mu(k) k^{-sigma}, logk[k-1] = log k (k = 1..n), shared.
    from zeta_spectral_gpu.dirac_mirror import mobius_sieve

    n, sigma = 64, 0.5
    packed = fam.pack_family(_chars([5, -3]), n, sigma=sigma)
    k = np.arange(1, n + 1, dtype=np.float64)
    np.testing.assert_allclose(
        packed.mu_amp, mobius_sieve(n)[1 : n + 1].astype(np.float64) * k**-sigma
    )
    np.testing.assert_allclose(packed.logk, np.log(k))


def test_locate_family_peaks_finds_beta_zeros():
    # FORWARD (CPU): the chi_-4 * mu partial sum |M'_z| peaks at the Dirichlet beta
    # zeros, located from the character alone. chi_-4 is member index 1 of FAMILY.
    chars = _chars(FAMILY)
    n = 4000
    grid = np.arange(2.0, 16.0, 0.01)
    block = fam.family_scan(chars, n, grid)
    peaks = fam.locate_family_peaks(np.abs(block), grid, n)
    beta_peaks = peaks[1]  # d = -4
    for known in (6.020949, 10.243770, 12.988056):
        assert beta_peaks.size and np.min(np.abs(beta_peaks - known)) < 0.05


def test_forward_no_zeros_consumed(monkeypatch):
    # Structural guarantee: the batched producer reads no zero. Poison both the
    # mpmath zero finders and the L-function zero scorer so any inverse-style
    # access explodes; the forward scan + locate must still run.
    import mpmath as mp

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("batched locator must not consume zeros (forward only)")

    monkeypatch.setattr(mp, "zetazero", _boom)
    monkeypatch.setattr(dl, "lfunction_zeros", _boom)
    grid = np.arange(2.0, 14.0, 0.02)
    block = fam.family_scan(_chars([-4, 5]), 2000, grid)
    peaks = fam.locate_family_peaks(np.abs(block), grid, 2000)
    assert len(peaks) == 2 and peaks[0].size  # beta zeros located, no zero read


# --- GPU agreement (the house GPU-vs-CPU rule) --------------------------------


def test_gpu_matches_cpu_family_real():
    # Decisive correctness: the batched real kernel reproduces the CPU reference
    # across the whole family and grid (bit-level fp64), small and larger n.
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import dirichlet_locator_family_gpu as gpu

    chars = _chars(FAMILY)
    grid = np.arange(0.5, 18.0, 0.05)
    for n in (300, 2000):
        g = gpu.family_scan_gpu(chars, n, grid)
        c = fam.family_scan(chars, n, grid)
        np.testing.assert_allclose(g, c, rtol=0, atol=1e-9)


def test_gpu_matches_cpu_family_complex():
    # The complex kernel: a mixed family with a genuinely complex character
    # (mod 5, j=1) still reproduces the CPU reference, including negative-E zeros.
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import dirichlet_locator_family_gpu as gpu

    chars = _chars(FAMILY) + [dl.dirichlet_character(5, 1)]
    assert not fam.pack_family(chars, 10).is_real
    grid = np.arange(-6.0, 18.0, 0.05)
    g = gpu.family_scan_gpu(chars, 1500, grid)
    c = fam.family_scan(chars, 1500, grid)
    np.testing.assert_allclose(g, c, rtol=0, atol=1e-9)


def test_gpu_preserves_shape_and_dtype():
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import dirichlet_locator_family_gpu as gpu

    chars = _chars(FAMILY)
    grid = np.linspace(2.0, 17.0, 53)
    out = gpu.family_scan_gpu(chars, 800, grid)
    assert out.shape == (len(chars), grid.size)
    assert out.dtype == np.complex128


def test_gpu_batched_reduces_to_single_character_kernel():
    # A one-member family must equal the established single-character GPU kernel
    # (dirac_mirror_gpu) -- ties the batched kernel to the validated one.
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import dirac_mirror_gpu as single
    from zeta_spectral_gpu import dirichlet_locator_family_gpu as gpu

    grid = np.arange(1.0, 16.0, 0.05)
    n = 1500
    for char in (ks.quadratic_character(-4), dl.dirichlet_character(5, 1)):
        w = dl.lfunction_weights(char, n)
        batched = gpu.family_scan_gpu([char], n, grid)[0]
        ref = single.mobius_partial_sum_gpu(grid, n, weights=w)
        np.testing.assert_allclose(batched, ref, rtol=0, atol=1e-9)


def test_gpu_locate_family_matches_cpu():
    # The forward producer end-to-end: batched GPU located peaks == CPU peaks.
    pytest.importorskip("cupy")
    grid = np.arange(0.5, 16.0, 0.01)
    gpu_peaks, backend = ks.locate_family_zeros(FAMILY, 4000, grid, prefer_gpu=True)
    assert backend == "gpu"
    cpu_peaks, _ = ks.locate_family_zeros(FAMILY, 4000, grid, prefer_gpu=False)
    assert len(gpu_peaks) == len(cpu_peaks) == len(FAMILY)
    for gp, cp in zip(gpu_peaks, cpu_peaks):
        assert gp.size == cp.size
        np.testing.assert_allclose(np.sort(gp), np.sort(cp), atol=2e-2)
