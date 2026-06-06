"""Invariants for the CCM convergence-law layer (issue #65).

Small ``N`` / modest ``dps`` so they stay CI-friendly. The decisive checks encode
the Phase-0 findings: (1) the genuine low-zero error is super-exponentially small
where fp64 reports an ``O(10)`` error (the precision-artifact finding); (2) the
resolution gate flags an under-resolved ``xi``; (3) on the resolved set the MAE
sits far below the Heisenberg floor (the floor is an edge phenomenon). The
full-scale study lives in ``scripts/run_ccm_convergence.py``.
"""

from __future__ import annotations

import mpmath as mp

from zeta_spectral_gpu import ccm_convergence as cc, plots


def test_heisenberg_bound_matches_formula():
    mp.mp.dps = 40
    lam = mp.sqrt(13)
    assert abs(cc.heisenberg_bound(lam) - 1 / (4 * mp.log(lam))) < 1e-35
    # In the cutoff variable x = lambda^2 it is 1/(2 ln x).
    assert abs(cc.heisenberg_bound(lam) - 1 / (2 * mp.log(13))) < 1e-35


def test_suggest_dps_increases_with_cutoff():
    assert cc.suggest_dps(12) >= 60  # floor
    assert cc.suggest_dps(40) > cc.suggest_dps(13)  # deeper cutoff needs more digits
    assert cc.suggest_dps(13) > cc.suggest_dps(12)


def test_convergence_errors_resolved_beats_floor():
    # N=40, x=13 at adequate precision: xi resolved (tiny first-zero error), and
    # the MAE over the tracked set sits FAR below the Heisenberg floor -- the
    # floor is a resolution-edge phenomenon, not a property of the tracked zeros.
    r = cc.convergence_errors(40, mp.sqrt(13), count=40, dps=110)
    assert r.resolved
    assert r.first_zero_error < mp.mpf(10) ** (-40)  # super-exp low-zero accuracy
    assert 0 < r.count <= 40
    assert r.mae < r.bound  # tracked-set MAE below the edge floor
    assert r.bound_ratio < 1


def test_resolution_gate_flags_underresolved():
    # Far too few digits to resolve xi (eps_N ~ 1e-59 >> 1e-25): the recovered
    # spectrum is garbage and the gate must catch it.
    r = cc.convergence_errors(40, mp.sqrt(13), count=20, dps=25)
    assert not r.resolved
    assert r.first_zero_error > mp.mpf("1e-3")


def test_fp64_corruption_genuine_below_fp64():
    # The headline: over the first low zeros the GENUINE (mpmath) error is
    # super-exponentially small, while the fp64 error is O(10) -- pure xi
    # corruption, not finite-cutoff error.
    corr = cc.fp64_spectrum_corruption(40, mp.sqrt(13), count=10, dps=110)
    assert corr.max_vs_zeros_mpmath < 1e-20  # genuine: super-exp small
    assert corr.max_vs_zeros_fp64 > 1.0  # fp64: O(1..10)
    # The fp64 "error" is explained by xi-corruption, not finite-cutoff error.
    assert corr.max_vs_mpmath > 1.0
    assert (
        corr.max_vs_zeros_fp64 <= corr.max_vs_mpmath + corr.max_vs_zeros_mpmath + 1e-6
    )


def test_accelerate_zero_reports_errors_and_is_forward():
    # A synthetic super-exponential cutoff-sequence converging to a known limit:
    # the raw last term is already excellent, so acceleration cannot beat it
    # (gain ~<= 1) -- the negative result, on a controlled sequence.
    mp.mp.dps = 50
    limit = mp.mpf("14.134725")
    seq = [limit + mp.mpf(10) ** (-3 * (j + 1)) for j in range(5)]  # super-exp
    az = cc.accelerate_zero(1, seq, [11, 12, 13, 14, 15], limit)
    assert az.raw_error == abs(seq[-1] - limit)
    # Output depends only on the input sequence + the (final-comparison) limit:
    # shifting the whole sequence shifts the raw error by the same amount.
    az2 = cc.accelerate_zero(1, [s + 7 for s in seq], [11, 12, 13, 14, 15], limit + 7)
    assert abs(az2.raw_error - az.raw_error) < mp.mpf(10) ** (-30)


def test_edge_figure_renders(tmp_path):
    study = {
        "x": 13,
        "N": 60,
        "errors": [1e-50, 1e-30, 1e-10, 1e-2, 0.3, 5.0],
        "bound": 0.195,
        "mae": 0.9,
        "k_cross": 5,
    }
    out = plots.ccm_convergence_edge_figure(study, out_path=tmp_path / "edge.png")
    assert out.exists() and out.stat().st_size > 0


def test_artifact_figure_renders(tmp_path):
    study = {
        "N": 80,
        "low": 12,
        "rows": [
            {
                "x": 11,
                "ln_lambda": 1.2,
                "genuine": 1e-24,
                "fp64": 10.8,
                "corruption": 10.8,
            },
            {
                "x": 13,
                "ln_lambda": 1.28,
                "genuine": 1e-34,
                "fp64": 13.7,
                "corruption": 13.7,
            },
            {
                "x": 15,
                "ln_lambda": 1.35,
                "genuine": 1e-44,
                "fp64": 23.5,
                "corruption": 23.5,
            },
        ],
    }
    out = plots.ccm_convergence_artifact_figure(
        study, out_path=tmp_path / "artifact.png"
    )
    assert out.exists() and out.stat().st_size > 0
