"""Invariants for the Seba / rank-one intermediate-statistics layer (issue #87).

Small ``N`` / modest ``dps`` so they stay CI-friendly. The decisive checks: (1)
the semi-Poisson reference constant ``<r~> = 1/2`` is reproduced by its defining
construction (every other level of a Poisson process); (2) on synthetic weak
couplings, both the first-order pinning formula ``delta_n = xi_n / R_n`` and the
local two-pole gap model reproduce the *exact* secular roots of
:func:`ccm.operator_eigenvalues`; (3) on a real (mpmath-resolved) CCM cell the
local model reproduces the measured gap occupancy and root positions, the tail
sits picket-ward of GUE, and the occupancy deficit plateau (the density
crossover) exists at an interior ordinate. The full-scale study lives in
``scripts/run_ccm_convergence.py --mode intermediate``.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import ccm, ccm_intermediate as ci, plots, spacing


def test_semi_poisson_rtilde_constant():
    # <r~> for semi-Poisson is exactly 1/2: consecutive spacings of the decimated
    # Poisson process are independent Gamma(2), and the folded mean of their ratio
    # integrates to 1/2. The simulated construction must reproduce it.
    rng = np.random.default_rng(87)
    levels = ci.semi_poisson_levels(200_000, rng)
    mean = float(np.nanmean(spacing.spacing_ratios(levels)))
    assert abs(mean - ci.MEAN_RATIO_SEMI_POISSON) < 5e-3
    # And it sits strictly between Poisson and GUE — "intermediate".
    assert spacing.MEAN_RATIO_POISSON < ci.MEAN_RATIO_SEMI_POISSON
    assert ci.MEAN_RATIO_SEMI_POISSON < spacing.MEAN_RATIO_GUE


def test_fold_even_couplings_slots_and_validation():
    N = 3
    xi = [7.0, 5.0, 3.0, 1.0, 3.0, 5.0, 7.0]  # even, index i -> n = i - N
    np.testing.assert_allclose(ci.fold_even_couplings(xi, N), [3.0, 5.0, 7.0])
    try:
        ci.fold_even_couplings(xi[:-1], N)
        raise AssertionError("length mismatch must raise")
    except ValueError:
        pass


def _synthetic_even_xi(N: int, eps: float, rng: np.random.Generator) -> list:
    """A unit dominant mode xi_0 plus weak signed couplings eps * O(1)."""
    g = rng.uniform(0.5, 1.0, size=N) * rng.choice([-1.0, 1.0], size=N)
    xi = np.zeros(2 * N + 1)
    xi[N] = 1.0
    xi[N + 1 :] = eps * g
    xi[:N] = (eps * g)[::-1]
    return [mp.mpf(v) for v in xi]


def test_pinning_formula_matches_exact_secular_roots():
    # Weak synthetic couplings: every root must sit at d_n + xi_n/R_n to second
    # order. Compare against the exact roots from ccm.operator_eigenvalues and
    # against the bare picket: the first-order offsets capture the jitter.
    mp.mp.dps = 30
    N, L, eps = 16, 1.8, 1e-4
    rng = np.random.default_rng(871)
    xi = _synthetic_even_xi(N, eps, rng)
    tail = ci.pinned_tail(xi, N, L)
    assert bool(tail.pinned.all())

    exact = np.array(
        [float(z) for z in ccm.operator_eigenvalues(xi, N, L, count=N)],
        dtype=np.float64,
    )
    delta = tail.spacing
    # Interior poles (the top pole's root can exceed d_N, outside the finder's
    # search range): match each prediction to its nearest exact root.
    for j in range(1, N - 1):
        z_pred = tail.poles[j] + tail.offsets[j]
        err = np.abs(exact - z_pred).min()
        picket_err = np.abs(exact - tail.poles[j]).min()
        assert err < 1e-3 * delta  # first-order accuracy ~ w^2 * Delta
        assert err < 0.05 * picket_err  # and far better than the bare picket


def test_local_gap_model_matches_exact_secular_roots():
    # The two-pole frozen-background model must reproduce the exact roots' gap
    # occupancy and positions on the same synthetic weak-coupling cell.
    mp.mp.dps = 30
    N, L, eps = 16, 1.8, 1e-4
    rng = np.random.default_rng(871)
    xi = _synthetic_even_xi(N, eps, rng)
    model = ci.local_gap_model(xi, N, L)
    exact = np.array(
        [float(z) for z in ccm.operator_eigenvalues(xi, N, L, count=N)],
        dtype=np.float64,
    )
    # Occupancy agrees with the exact root histogram on every gap the finder
    # covers (it stops at d_N, so drop predictions in the top gap if the exact
    # list ran short).
    occ_exact = np.histogram(exact, bins=model.edges)[0]
    agree = model.occupancy[:-1] == occ_exact[:-1]
    assert agree.all()
    # Positions: every exact root has a predicted root within ~w^2 of the gap.
    for z in exact:
        assert np.abs(model.levels - z).min() < 1e-2 * model.spacing


def test_deficit_plateau_reads_density_crossover():
    # Synthetic occupancy: deficit grows over the first four gaps (0s), peaks,
    # and is paid back by a late 2 — the plateau must span peak-first..peak-last.
    m = ci.LocalGapModel(
        N=8,
        L=2.0,
        edges=np.pi * np.arange(0, 9),
        occupancy=np.array([0, 0, 1, 0, 1, 1, 2, 1]),
        levels=np.array([]),
    )
    p = ci.deficit_plateau(m)
    assert p is not None
    assert p.deficit_max == 3
    assert p.t_first == float(m.edges[4])  # deficit hits 3 after gap index 3
    assert p.t_last == float(m.edges[6])  # ... and stays 3 through gap index 5
    assert p.t_mid == 0.5 * (p.t_first + p.t_last)
    # No deficit -> None.
    m2 = ci.LocalGapModel(
        N=3,
        L=2.0,
        edges=np.pi * np.arange(0, 4),
        occupancy=np.array([1, 1, 1]),
        levels=np.array([]),
    )
    assert ci.deficit_plateau(m2) is None


def test_windowed_rtilde_picket_and_semi_poisson():
    # A rigid picket gives r~ = 1 in every window; a semi-Poisson spectrum sits
    # near 1/2. Window centers are level values, ascending.
    picket = np.arange(1.0, 401.0)
    rows = ci.windowed_rtilde(picket, window=50, step=25)
    assert rows.shape[0] > 3
    np.testing.assert_allclose(rows[:, 1], 1.0, atol=1e-12)
    assert np.all(np.diff(rows[:, 0]) > 0)

    rng = np.random.default_rng(872)
    sp = ci.semi_poisson_levels(6_000, rng)
    rows = ci.windowed_rtilde(sp, window=1_500, step=750)
    assert np.all(np.abs(rows[:, 1] - 0.5) < 0.05)
    # Degenerate input: too few levels -> empty, not an error.
    assert ci.windowed_rtilde(picket[:10], window=50, step=10).size == 0


def test_real_ccm_cell_locality_and_intermediate_tail():
    # A real (mpmath-resolved) small cell: x = 6, N = 24. The local two-pole
    # model must reproduce the measured spectrum's gap occupancy and positions,
    # the deficit plateau must exist at an interior ordinate, and the tail must
    # sit picket-ward of GUE (the intermediate regime) with the local model's
    # tail statistics close to the measured ones.
    N, dps = 24, 60
    lam = mp.sqrt(6)
    with mp.workdps(dps):
        L = 2 * mp.log(lam)
        A = ccm.assemble_weil_matrix(N, lam)
        mode = ccm.smallest_even_eigenvector(A, N)
        spec = ccm.operator_eigenvalues(mode.eigenvector, N, L, count=N)
        model = ci.local_gap_model(mode.eigenvector, N, L)
        tail = ci.pinned_tail(mode.eigenvector, N, L)
    spec_f = np.sort(np.array([float(s) for s in spec]))

    occ_meas = np.histogram(spec_f, bins=model.edges)[0]
    assert float(np.mean(model.occupancy == occ_meas)) > 0.85
    near = np.array([np.abs(spec_f - z).min() for z in model.levels])
    assert float(np.median(near)) < 0.1 * model.spacing

    p = ci.deficit_plateau(model)
    assert p is not None and 0 < p.t_first <= p.t_mid <= p.t_last
    assert p.t_last < float(model.edges[-1])

    # The effective coupling is O(1)-intermediate in the tail (not w -> 0), and
    # the upper-half statistics sit between GUE and the picket.
    upper = spec_f[spec_f > float(model.edges[N // 2])]
    rt_meas = float(np.nanmean(spacing.spacing_ratios(upper)))
    assert spacing.MEAN_RATIO_GUE < rt_meas < 1.0
    upper_pred = model.levels[model.levels > float(model.edges[N // 2])]
    rt_pred = float(np.nanmean(spacing.spacing_ratios(upper_pred)))
    assert abs(rt_pred - rt_meas) < 0.1
    w_upper = tail.w[N // 2 :]
    assert 0.01 < float(np.median(w_upper)) < 2.0


def test_intermediate_stats_figure_renders(tmp_path):
    study = {
        "N": 160,
        "window": 40,
        "rows": [
            {
                "x": 6,
                "t_star": 60.8,
                "t_dens_first": 24.5,
                "t_dens": 40.3,
                "t_dens_last": 56.1,
                "deficit_max": 5,
                "rtilde_tail_meas": 0.82,
                "rtilde_tail_pred": 0.80,
                "poles": [10.0, 20.0, 30.0],
                "w": [1.2, 0.3, 0.4],
                "windowed_meas": [[30.0, 0.62], [80.0, 0.8], [200.0, 0.85]],
                "windowed_pred": [[80.0, 0.82], [200.0, 0.84]],
            },
            {
                "x": 14,
                "t_star": 167.2,
                "t_dens_first": 64.3,
                "t_dens": 89.3,
                "t_dens_last": 114.3,
                "deficit_max": 13,
                "rtilde_tail_meas": 0.77,
                "rtilde_tail_pred": 0.74,
                "poles": [10.0, 20.0, 30.0],
                "w": [2.0, 0.6, 0.5],
                "windowed_meas": [[30.0, 0.6], [120.0, 0.63], [250.0, 0.78]],
                "windowed_pred": [[200.0, 0.74], [250.0, 0.76]],
            },
        ],
        "hist_edges": [0.0, 0.5, 1.0, 1.5, 2.0],
        "hist_meas": [0.1, 0.7, 0.15, 0.05],
        "hist_pred": [0.08, 0.72, 0.16, 0.04],
    }
    out = plots.ccm_intermediate_stats_figure(study, out_path=tmp_path / "int.png")
    assert out.exists() and out.stat().st_size > 0
