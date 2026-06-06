"""Invariants for the De Bruijn–Newman forward flow spike (#20).

The decisive test is ``test_forward_h0_roots_are_doubled_ordinates``: it confirms
the forward generator (root-finding ``H_t`` built from the structural kernel
``Phi``) reproduces the known ordinates at ``t = 0`` *and* pins the
``z = 2 gamma`` normalisation — without ever feeding the zeros in. These run on
CPU at modest precision so they stay CI-friendly; the scientific rigidity-vs-``t``
readout lives in ``scripts/run_debruijn_newman.py``.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import debruijn_newman as dbn


def test_phi_is_positive_and_series_converges():
    mp.mp.dps = 30
    # Phi(0) is a known constant; Phi > 0 and decays super-exponentially.
    assert abs(float(dbn.phi(0)) - 0.4466969) < 1e-6
    vals = [float(dbn.phi(u)) for u in (0.0, 0.2, 0.4, 0.6)]
    assert all(v > 0 for v in vals)
    assert vals == sorted(vals, reverse=True)  # monotone decreasing on [0, 0.6]
    # Truncating the series hardly moves the value (it has converged).
    near = dbn.phi(0.1, max_terms=3)
    full = dbn.phi(0.1, max_terms=200)
    assert abs(float(near - full)) < 1e-20


def test_h0_integral_matches_closed_form():
    # The forward integral H_t(z, t=0) must equal the closed form xi(1/2+iz/2)/8.
    mp.mp.dps = 30
    for z in (5.0, 20.0, 40.0):
        a = dbn.h_t(z, 0)
        b = dbn.h0_reference(z)
        assert abs(float(a - b)) <= 1e-12 * (abs(float(b)) + 1e-30)


def test_forward_h0_roots_are_doubled_ordinates():
    # The crux: forward-generated H_0 zeros sit at z = 2*gamma (gamma = ordinate).
    # Confirms correctness and the normalisation, with zeros only ever an output.
    n = 4
    # scan_step 1.0 is safe (the first H_0 zero gaps are >= ~5 in z) and keeps
    # the (extended-precision) root-finding cheap enough for CI.
    roots = dbn.h_t_zeros(0.0, n, dps=30, scan_step=1.0, z_lo=20.0, cache_dir=None)
    gammas = np.array([float(mp.zetazero(k).imag) for k in range(1, n + 1)])
    np.testing.assert_allclose(roots, 2.0 * gammas, atol=1e-7)


def test_flow_keeps_zeros_real_ordered_and_moves_them():
    # For t > 0 (and RH-true heights) H_t has n real, ascending, positive zeros,
    # displaced from the t = 0 set by the heat flow.
    n = 5
    z0 = dbn.h_t_zeros(0.0, n, dps=30, scan_step=1.0, z_lo=20.0, cache_dir=None)
    z1 = dbn.h_t_zeros(1.0, n, dps=30, scan_step=1.0, z_lo=20.0, cache_dir=None)
    assert z1.shape == (n,)
    assert np.all(np.isfinite(z1)) and np.all(z1 > 0)
    assert np.all(np.diff(z1) > 0)  # ascending, no collisions
    assert np.max(np.abs(z1 - z0)) > 1e-3  # the flow actually moved them


def test_empirical_unfold_gives_unit_mean_spacing():
    # A non-uniform (quadratic-density) level set unfolds to unit mean spacing.
    rng = np.random.default_rng(0)
    counts = np.arange(1, 401)
    z = np.sqrt(counts) * 20.0  # density ~ z, a smooth non-uniform staircase
    z = np.sort(z + rng.normal(scale=0.05, size=z.size))
    w = dbn.empirical_unfold(z, degree=6)
    s = np.diff(w)
    assert abs(s.mean() - 1.0) < 0.05
    assert np.all(np.diff(w) > 0)  # unfolding preserves order


def test_ordinate_and_dps_helpers_scale_with_height():
    # _ordinate_for_index is increasing and near the right heights (it slightly
    # overestimates by design, to size the scan/precision conservatively).
    g10 = dbn._ordinate_for_index(10)
    g50 = dbn._ordinate_for_index(50)
    assert 45.0 < g10 < 58.0 < g50
    # required_dps grows with the top height (more cancellation to outrun).
    assert dbn.required_dps(50) > dbn.required_dps(5) > 0
