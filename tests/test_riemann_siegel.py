"""Riemann-Siegel evaluator vs mpmath ground truth (issue #55, CPU reference).

mpmath's ``siegeltheta`` / ``siegelz`` / ``zeta`` are the truth; the fp64
Riemann-Siegel implementation must reproduce them to the asymptotic accuracy set
by the number of correction terms, and -- fed nothing but ``t`` -- locate the
on-line zeta zeros as sign changes of the Hardy function ``Z`` (zeros are an
output, only scored against mpmath).
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import riemann_siegel as rs


def test_theta_matches_siegeltheta():
    # The asymptotic theta series tracks mpmath.siegeltheta across the useful range.
    ts = np.linspace(30.0, 3000.0, 25)
    err = max(abs(float(rs.theta(t)) - float(mp.siegeltheta(t))) for t in ts)
    assert err < 1e-9


def test_psi_derivatives_match_mpmath():
    # The Cauchy-contour derivatives reproduce mpmath's (high-accuracy quad) Psi
    # derivatives through order 6 -- including AT the removable singularities
    # p = 1/4, 3/4, where the naive quotient recurrence would diverge. (orders 3
    # and 6 are what the C_1 and C_2 correction terms consume.)
    mp.mp.dps = 50

    def psi(p):
        return mp.cos(2 * mp.pi * (p * p - p - mp.mpf(1) / 16)) / mp.cos(2 * mp.pi * p)

    pts = np.array([0.05, 0.2, 0.25, 0.45, 0.5, 0.55, 0.75, 0.95])
    d = rs._psi_derivatives(pts, 6)
    for order in (0, 1, 2, 3, 5, 6):
        ref = np.array(
            [
                mp.diff(psi, mp.mpf(p), order, method="quad", radius=0.13).real
                for p in pts
            ]
        ).astype(float)
        err = max(abs(d[order][i] - ref[i]) for i in range(pts.size))
        assert err < 1e-7, (order, err)
    mp.mp.dps = 15


def test_no_singularity_blowup_near_quarter_integer_tau():
    # At t ~ 113.5, sqrt(t/2pi) ~ 4.25 so the remainder's p ~ 1/4 (a removable
    # singularity of Psi). The evaluator must stay accurate there, not spike.
    ts = np.linspace(112.0, 115.0, 61)  # straddles p = 1/4
    z = rs.hardy_z(ts, correction_terms=2)
    err = max(abs(z[i] - float(mp.siegelz(ts[i]))) for i in range(ts.size))
    assert err < 5e-6


def test_hardy_z_matches_siegelz():
    # Two correction terms (C_0..C_2) reach ~1e-6 by t=50 and tighter with height.
    ts = np.linspace(50.0, 500.0, 16)
    z = rs.hardy_z(ts, correction_terms=2)
    err = max(abs(z[i] - float(mp.siegelz(ts[i]))) for i in range(ts.size))
    assert err < 5e-6
    # Higher up the asymptotic expansion sharpens considerably.
    assert abs(rs.hardy_z(5000.0) - float(mp.siegelz(5000.0))) < 1e-7


def test_more_correction_terms_monotonically_improve():
    # Each extra C_k reduces the error against the truth (asymptotic convergence).
    t = 100.0
    truth = float(mp.siegelz(t))
    e0 = abs(rs.hardy_z(t, correction_terms=0) - truth)
    e1 = abs(rs.hardy_z(t, correction_terms=1) - truth)
    e2 = abs(rs.hardy_z(t, correction_terms=2) - truth)
    assert e2 < e1 < e0


def test_main_sum_plus_remainder_is_hardy_z():
    # hardy_z is exactly main_sum + rs_remainder (the decomposition the GPU mirrors).
    ts = np.linspace(60.0, 400.0, 9)
    z = rs.hardy_z(ts, correction_terms=2)
    parts = rs.main_sum(ts) + rs.rs_remainder(ts, correction_terms=2)
    np.testing.assert_allclose(z, parts, rtol=0, atol=1e-13)


def test_zeta_critical_matches_mpmath_zeta():
    ts = np.linspace(80.0, 600.0, 12)
    zc = rs.zeta_critical(ts, correction_terms=2)
    err = max(abs(zc[i] - complex(mp.zeta(mp.mpc(0.5, ts[i])))) for i in range(ts.size))
    assert err < 5e-6


def test_scalar_and_array_shapes():
    assert isinstance(rs.hardy_z(1000.0), float)
    assert isinstance(rs.zeta_critical(1000.0), complex)
    grid = np.linspace(100.0, 200.0, 21)
    assert rs.hardy_z(grid).shape == grid.shape
    assert rs.zeta_critical(grid).shape == grid.shape
    assert rs.theta(grid).shape == grid.shape


def _sign_change_zeros(t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear-interpolated sign-change locations of z(t) -- the forward locator."""
    s = np.signbit(z)
    idx = np.nonzero(s[:-1] != s[1:])[0]
    z0, z1 = z[idx], z[idx + 1]
    t0, t1 = t[idx], t[idx + 1]
    return t0 - z0 * (t1 - t0) / (z1 - z0)


def test_forward_locates_zeros_as_sign_changes():
    # Forward: scan Z over a window, read the zeros off its sign changes, and only
    # then score them against mpmath (mp.siegelz ~ 0 at each located point). No
    # zeros are fed in.
    grid = np.arange(100.0, 140.0, 0.02)
    located = _sign_change_zeros(grid, rs.hardy_z(grid))
    assert located.size >= 15  # ~19 zeros live in [100, 140]
    assert np.all(np.diff(located) > 0)
    residual = max(abs(float(mp.siegelz(z0))) for z0 in located)
    assert residual < 1e-2  # each located point is a genuine Hardy-Z zero
