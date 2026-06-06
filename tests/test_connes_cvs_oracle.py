"""Baseline-oracle invariants (#16).

The checked-in fixture ``tests/fixtures/connes_cvs_c13_oracle.json`` is the
known-good ``lambda_min`` for the connes-cvs ``c=13, N=80, T=400, dps=80`` cell.
These tests guard the fixture and the loader; the always-on ones need only
``mpmath`` (a core dep), so they run in CI where ``connes-cvs`` is absent. The
actual reproduction (rebuilding the cell with connes-cvs) is a ``slow`` test,
deselected by default and skipped cleanly when the ``oracle`` extra isn't
installed.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from zeta_spectral_gpu import oracle

# First nontrivial zeta ordinate, t_1, to ample precision (independent of the
# fixture). Used to confirm the recorded detected zero and its error are honest.
GAMMA1 = "14.134725141734693790457251983562470270784257115699243175685567460149"


def test_fixture_matches_documented_cell():
    rec = oracle.load_connes_cvs_oracle()
    assert rec["cell"] == {"c": 13, "N": 80, "T": 400, "dps": 80}
    with mp.workdps(90):
        lam = mp.mpf(rec["lambda_min"])
        target = mp.mpf("2.528266e-59")
        assert abs(lam - target) / target < mp.mpf("1e-5")


def test_loader_preserves_high_precision():
    # Far more than fp64's ~16 digits must survive the round-trip.
    lam = oracle.lambda_min()
    assert isinstance(lam, mp.mpf)
    with mp.workdps(90):
        assert mp.mpf("2.5282661401e-59") < lam < mp.mpf("2.5282661403e-59")
    # The recorded string itself carries the full ~80-digit value.
    assert len(oracle.load_connes_cvs_oracle()["lambda_min"].split("e")[0]) > 50


def test_detected_zero_and_error_are_honest():
    # Independent forward check: the detected zero must match the true ordinate
    # t_1, and the recorded |gamma_1 - t_1| must equal that gap. Uses mpmath's
    # t_1 (a constant here), never feeding zeros into any operator.
    rec = oracle.load_connes_cvs_oracle()
    with mp.workdps(90):
        detected = mp.mpf(rec["gamma1_detected"])
        recorded_err = mp.mpf(rec["gamma1_abs_error"])
        gap = abs(detected - mp.mpf(GAMMA1))
        # detected agrees with t_1 to ~54 digits; the gap is ~1.77e-55.
        assert mp.mpf("1e-55") < gap < mp.mpf("3e-55")
        assert abs(gap - recorded_err) / recorded_err < mp.mpf("1e-4")


def test_published_normalization_factor_is_in_band():
    # The ~1.2-1.3 convention factor: connes-cvs's reproduce-paper c=13 first-zero
    # error (2.005e-55) vs the published values it's compared against.
    rec = oracle.load_connes_cvs_oracle()
    published = rec["normalization_note"]["published_gamma1_err"]
    connes_cvs_repro = mp.mpf("2.005e-55")
    for value in published.values():
        factor = mp.mpf(value) / connes_cvs_repro
        assert mp.mpf("1.15") < factor < mp.mpf("1.35")


@pytest.mark.slow
def test_connes_cvs_reproduces_oracle():
    # The heavy check: rebuild the exact cell and confirm lambda_min matches the
    # fixture to high precision. ~minutes; needs `uv sync --extra oracle`.
    cc = pytest.importorskip("connes_cvs")
    rec = oracle.load_connes_cvs_oracle()
    cell = rec["cell"]
    mp.mp.dps = cell["dps"]
    q = cc.build_galerkin_matrix(c=cell["c"], N=cell["N"], T=cell["T"], dps=cell["dps"])
    lam, _ = cc.compute_ground_state(q)
    expected = oracle.lambda_min(dps=cell["dps"] + 5)
    assert abs(lam - expected) / expected < mp.mpf(f"1e-{cell['dps'] - 10}")
