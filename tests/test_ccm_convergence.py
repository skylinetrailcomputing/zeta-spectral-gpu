"""Invariants for the CCM convergence-law layer (issue #65).

Small ``N`` / modest ``dps`` so they stay CI-friendly. The decisive checks encode
the Phase-0 findings: (1) the genuine low-zero error is super-exponentially small
where fp64 reports an ``O(10)`` error (the precision-artifact finding); (2) the
resolution gate flags an under-resolved ``xi``; (3) on the resolved set the MAE
sits far below the Heisenberg floor (the floor is an edge phenomenon). The
full-scale study lives in ``scripts/run_ccm_convergence.py``.
"""

from __future__ import annotations

import math

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


def test_tracking_length_counts_leading_tracked_block():
    # Four levels tracked to <1e-3 rel, then a sharp detach. k* is the contiguous
    # leading block length, and it is monotone non-decreasing in rel_tol.
    zeros = [14.0, 21.0, 25.0, 30.0, 32.0, 37.0]
    errors = [1e-50, 1e-30, 1e-10, 3e-4, 0.5, 5.0]  # rel at idx3 = 1e-5, idx4 ~ 1.6e-2
    assert cc.tracking_length(errors, zeros, rel_tol=1e-3) == 4
    assert cc.tracking_length(errors, zeros, rel_tol=1e-6) == 3  # idx3 (1e-5) drops
    # A lone spike *past* the break does not extend the block (contiguous, not "most").
    assert (
        cc.tracking_length([1e-50, 9.9, 1e-50], [14.0, 21.0, 25.0], rel_tol=1e-3) == 1
    )


def test_tracking_length_zero_when_first_detaches():
    # An under-resolved xi: even nu_1 is off -> k* = 0, never reported as signal.
    assert cc.tracking_length([1.0, 1e-50], [14.0, 21.0], rel_tol=1e-3) == 0


def test_tracking_height_picks_edge_ordinate():
    zeros = [14.0, 21.0, 25.0, 30.0]
    assert cc.tracking_height(zeros, 3) == 25.0
    assert cc.tracking_height(zeros, 0) is None


def test_tracking_length_resolved_cell_monotone_and_method_agrees():
    # On a genuinely resolved cell the tracked block is non-trivial, bounded by the
    # compared count, and looser tol tracks at least as far as tighter tol. The
    # dataclass method must agree with the free function (mpf inputs exercised here).
    r = cc.convergence_errors(40, mp.sqrt(13), count=40, dps=110)
    assert r.resolved
    k_loose = cc.tracking_length(r.errors, r.zeros, rel_tol=1e-2)
    k_tight = cc.tracking_length(r.errors, r.zeros, rel_tol=1e-8)
    assert 0 < k_tight <= k_loose <= r.count
    assert r.tracking_length(rel_tol=1e-2) == k_loose


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


def test_gain_law_tracks_log_window_not_prime_content():
    # #94: forward reproduction of Groskin's connes-cvs observation -- the per-step
    # first-zero gain is governed by the log-window growth (ln c), not prime content.
    # Small N / short integer sweep keeps it CI-friendly but decisive.
    law = cc.first_zero_gain_law([11, 12, 13, 14, 15, 16], N=48)
    assert all(law.resolved)  # every cell's xi resolved at the suggested dps
    assert len(law.steps) == 5

    # The headline step: c=13 -> 14 adds NO new prime and NO new prime power, yet the
    # first-zero error still drops multiple orders of magnitude.
    step = next(s for s in law.steps if (s.c0, s.c1) == (13, 14))
    assert step.dpi == 0 and step.dpp == 0
    assert step.gain > 2.5

    # The gain is far better explained by the log-window than by arithmetic content:
    # strong (negative) correlation with ln c, ~zero with the new-prime count.
    assert abs(law.r_gain_vs_ln_c) > 0.7
    assert abs(law.r_gain_vs_ln_c) > abs(law.r_gain_vs_dpi)
    assert abs(law.r_gain_vs_dln_c) > abs(law.r_gain_vs_dpp)


def test_gain_law_single_cutoff_has_no_steps():
    # A one-cutoff sweep forms no steps, so every correlation is nan (undefined) --
    # the same guard that protects an all-unresolved sweep from a spurious r.
    law = cc.first_zero_gain_law([13], N=48)
    assert law.resolved == [True]
    assert law.steps == []
    assert math.isnan(law.r_gain_vs_ln_c)
    assert math.isnan(law.r_gain_vs_dpi)


def test_safe_corr_guards_degenerate_input():
    # Fewer than two points, or a constant axis (e.g. every step adds one prime),
    # has no defined correlation -- must be nan, never a spurious +/-1.
    assert math.isnan(cc._safe_corr([1.0], [2.0]))
    assert math.isnan(cc._safe_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert abs(cc._safe_corr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-12


def test_prime_counts_match_known_values():
    # pi(c) and the von Mangoldt support count, the two "prime content" axes.
    assert cc.prime_count(13) == 6  # 2,3,5,7,11,13
    assert cc.prime_count(14) == 6  # no prime in (13, 14] -- the no-new-prime step
    assert cc.prime_power_count(13) == 9  # +4,8,9 over the 6 primes
    assert cc.prime_power_count(14) == 9  # 14 is not a prime power


def test_edge_corruption_confined_to_low_band():
    # #82: fp64 xi-corruption is concentrated in the low / near-null band; the edge
    # eigenvalues are pinned to the bulk poles d_n and robust. N must be large enough
    # that the resolution edge is actually reached (k_floor is a real index).
    prof = cc.edge_corruption_profile(80, mp.sqrt(13), dps=70)
    assert prof.k_floor is not None and prof.count > prof.k_floor  # edge is reached
    s = cc.summarize_edge_bands(prof)
    # The corruption is larger in the low band than at the edge -- confined to the
    # near-null xi-direction, not the edge.
    assert s["edge"]["max_corruption"] < s["low"]["max_corruption"]
    # In the low band the genuine error is super-exp small while fp64 is O(1): the
    # fp64 low spectrum is *entirely* xi-corruption.
    assert s["low"]["max_genuine"] < 1.0
    assert s["low"]["max_corruption"] > 1.0

    # Near the edge the fp64 and mpmath roots converge to the SAME pole gap
    # (pole-pinned), whereas the low band is gap-scrambled -- the robustness
    # signature. Compare the mean gap mismatch head vs tail.
    def mean_gap_diff(sl):
        gm, gf = prof.gap_mpmath[sl], prof.gap_fp64[sl]
        return sum(abs(a - b) for a, b in zip(gm, gf)) / max(1, len(gm))

    assert mean_gap_diff(slice(prof.count - 8, prof.count)) < mean_gap_diff(slice(0, 8))


def test_summarize_edge_bands_crossover_readout():
    # The crossover readout on synthetic bands: E = max_k |nu_k - zeta_k| is set by
    # whichever is larger -- the bounded low-band corruption (small N) or the genuine
    # resolution edge (large N). E_argmax_band flips low -> edge across the crossover.
    common = dict(
        N=10,
        cutoff=mp.mpf(13),
        dps=50,
        count=6,
        corruption=[10.0, 9.0, 8.0, 2.9, 2.0, 0.5],
        gap_mpmath=[5, 7, 9, 10, 11, 12],
        gap_fp64=[2, 4, 6, 9, 11, 12],
        bound=0.2,
        k_floor=3,
    )
    # case A: genuine edge (5) < low-band corruption (10) -> E is low-band (corrupted)
    a = cc.EdgeCorruption(
        genuine=[1e-30, 1e-20, 1e-10, 0.3, 2.0, 5.0],
        fp64_error=[10.0, 9.0, 8.0, 3.0, 4.0, 5.2],
        **common,
    )
    sa = cc.summarize_edge_bands(a)
    assert sa["k_split"] == 3
    assert sa["low"]["max_corruption"] == 10.0
    assert sa["E_argmax_band"] == "low"
    # case B: genuine edge grows to 30 -> the (robust) edge sets E -> edge-dominated
    b = cc.EdgeCorruption(
        genuine=[1e-30, 1e-20, 1e-10, 0.3, 15.0, 30.0],
        fp64_error=[10.0, 9.0, 8.0, 3.0, 16.0, 30.5],
        **common,
    )
    sb = cc.summarize_edge_bands(b)
    assert sb["E_argmax_band"] == "edge"
    assert sb["edge"]["max_genuine"] == 30.0


def test_edge_corruption_figure_renders(tmp_path):
    study = {
        "x": 13,
        "crossover_N": 120,
        "edge_robust": True,
        "rows": [
            {
                "N": 80,
                "low_corruption": 13.7,
                "edge_corruption": 3.8,
                "edge_genuine": 6.0,
                "E_fp64": 13.7,
                "E_genuine": 6.0,
                "E_band": "low",
            },
            {
                "N": 120,
                "low_corruption": 19.9,
                "edge_corruption": 9.0,
                "edge_genuine": 30.1,
                "E_fp64": 28.8,
                "E_genuine": 30.1,
                "E_band": "edge",
            },
            {
                "N": 160,
                "low_corruption": 12.6,
                "edge_corruption": 8.2,
                "edge_genuine": 58.1,
                "E_fp64": 65.4,
                "E_genuine": 58.1,
                "E_band": "edge",
            },
        ],
        "focus": {
            "N": 160,
            "count": 6,
            "k_floor": 3,
            "bound": 0.195,
            "genuine": [1e-50, 1e-30, 1e-10, 0.3, 5.0, 30.0],
            "fp64_error": [10.0, 9.0, 8.0, 3.0, 6.0, 30.0],
            "corruption": [10.0, 9.0, 8.0, 2.9, 2.0, 0.5],
            "gap_mpmath": [5, 7, 9, 10, 11, 12],
            "gap_fp64": [2, 4, 6, 9, 11, 12],
        },
    }
    out = plots.ccm_edge_corruption_figure(study, out_path=tmp_path / "ec.png")
    assert out.exists() and out.stat().st_size > 0


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


def test_tracking_range_figure_renders(tmp_path):
    study = {
        "N": 160,
        "rel_tol": 1e-3,
        "c_hat": 11.75,
        "rows": [
            {"x": 6, "k_star": 14, "ratio": 10.1, "capped": False, "k_pred": 13},
            {"x": 12, "k_star": 48, "ratio": 11.6, "capped": False, "k_pred": 47},
            {"x": 22, "k_star": 117, "ratio": 12.1, "capped": False, "k_pred": 115},
            {"x": 30, "k_star": 160, "ratio": 8.0, "capped": True, "k_pred": 200},
        ],
    }
    out = plots.ccm_tracking_range_figure(study, out_path=tmp_path / "track.png")
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
