"""Reusable Dirichlet-``L`` locator core (issue #60).

The packaging layer over the forward locator (#25/#42): :mod:`dirichlet_locator`
factors the scan -> peaks -> score pipeline the CLI used to inline. The fast tests
pin the pure helpers and the orchestration with *injected* ground-truth zeros (no
mpmath, so they run in CI); the ``slow`` test exercises the full default path that
computes the comparison zeros itself. Forward throughout: the comparison zeros only
score the located peaks — they never enter the scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import dirichlet as dl
from zeta_spectral_gpu import dirichlet_locator as dlc

# Known low ordinates of the Dirichlet beta L-function (mod 4 odd character).
BETA_ZEROS = np.array([6.020949, 10.243770, 12.988056])


# --- pure helpers -------------------------------------------------------------


def test_peak_threshold_tracks_log_n():
    assert dlc.peak_threshold(8000) == pytest.approx(0.3 * np.log(8000))


def test_local_maxima_finds_interior_peaks_above_height():
    abs_m = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0])
    idx = dlc.local_maxima(abs_m, height=0.5)
    np.testing.assert_array_equal(idx, [1, 3, 6])
    # Raising the bar drops the smaller maxima.
    np.testing.assert_array_equal(dlc.local_maxima(abs_m, height=2.5), [6])


def test_match_peaks_pairs_nearest_within_tolerance():
    peaks = np.array([6.0, 10.2])
    matches = dlc.match_peaks(peaks, BETA_ZEROS, tol=0.3)
    assert [m.matched for m in matches] == [True, True, False]
    assert matches[0].peak_E == 6.0 and matches[0].error == pytest.approx(0.020949)
    assert matches[2].peak_E is None and matches[2].error is None


def test_score_scan_requires_zeros_or_character():
    with pytest.raises(ValueError):
        dlc.score_scan(np.arange(3.0), np.ones(3), height=0.1)


# --- orchestration (fast: injected zeros, CPU scan) ---------------------------


def test_locate_and_score_real_character_cpu():
    # FORWARD: feed only chi (mod 4) + mu; the chi*mu partial sum locates the
    # injected beta zeros. Injecting true_zeros keeps it mpmath-free (fast).
    chi4 = dl.dirichlet_character(4, 1)
    grid = np.arange(2.0, 15.0, 0.01)
    res = dlc.locate_and_score(
        chi4, 4000, grid, prefer_gpu=False, true_zeros=BETA_ZEROS
    )
    assert res.backend == "cpu"
    assert res.is_real and res.kernel == "mobius_locator"
    assert res.peaks.size >= BETA_ZEROS.size
    assert res.matched_count == BETA_ZEROS.size
    assert all(m.error < 0.1 for m in res.matches)
    assert res.height == pytest.approx(dlc.peak_threshold(4000))


def test_locate_and_score_complex_character_uses_weighted_path():
    # A genuinely complex character (mod 5): complex chi*mu weights select the
    # weighted_locator path; the scan still produces well-formed peaks. (Scoring
    # the complex zeros end-to-end is the slow mpmath test in test_dirichlet.)
    chi5 = dl.dirichlet_character(5, 1)
    grid = np.arange(-12.0, 12.0, 0.01)
    res = dlc.locate_and_score(
        chi5, 5000, grid, prefer_gpu=False, true_zeros=np.array([])
    )
    assert not res.is_real and res.kernel == "weighted_locator"
    assert res.abs_m.shape == grid.shape
    assert res.peaks.size > 0


def test_scan_locator_matches_mobius_partial_sum_cpu():
    from zeta_spectral_gpu import dirac_mirror as dm

    chi4 = dl.dirichlet_character(4, 1)
    grid = np.arange(2.0, 8.0, 0.05)
    values, backend = dlc.scan_locator(chi4, 2000, grid, prefer_gpu=False)
    assert backend == "cpu"
    w = dl.lfunction_weights(chi4, 2000)
    ref = dm.mobius_partial_sum(grid, 2000, weights=w)
    np.testing.assert_allclose(values, ref, rtol=0, atol=1e-12)


def test_modulus_one_trivial_character_is_zeta_path():
    # The period-1 trivial character reproduces zeta: real weights == mu, so the
    # fast real locator is selected (the CLI's --modulus 1 = zeta route).
    trivial = np.ones(1, dtype=np.complex128)
    grid = np.arange(10.0, 16.0, 0.01)
    res = dlc.locate_and_score(
        trivial, 4000, grid, prefer_gpu=False, true_zeros=np.array([14.134725])
    )
    assert res.is_real and res.kernel == "mobius_locator"
    assert res.matched_count == 1
    assert res.matches[0].error < 0.1


# --- full default path (slow: independent mpmath zeros) -----------------------


@pytest.mark.slow
def test_locate_and_score_computes_zeros_end_to_end():
    chi4 = dl.dirichlet_character(4, 1)
    grid = np.arange(2.0, 14.0, 0.01)
    res = dlc.locate_and_score(chi4, 4000, grid, prefer_gpu=False)
    assert res.true_zeros.size >= 3  # zeros found independently over the window
    assert res.matched_count >= res.true_zeros.size - 1  # essentially all located


# --- GPU twin (skipped without cupy) ------------------------------------------


def test_scan_locator_gpu_matches_cpu():
    pytest.importorskip("cupy")
    chi4 = dl.dirichlet_character(4, 1)
    grid = np.arange(2.0, 14.0, 0.01)
    gpu_vals, backend = dlc.scan_locator(chi4, 4000, grid, prefer_gpu=True)
    assert backend == "gpu"
    cpu_vals, _ = dlc.scan_locator(chi4, 4000, grid, prefer_gpu=False)
    assert float(np.max(np.abs(gpu_vals - cpu_vals))) < 1e-10
