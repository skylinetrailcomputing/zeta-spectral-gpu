"""Correctness of the sequence accelerators (issue #65), against known limits.

Each transform is checked to (a) recover the analytic limit far better than the
raw sequence and (b) work in both float and mpmath arithmetic. The forward
litmus is structural: the accelerators take only a numeric sequence, so they
track the *input's* limit and cannot consume an external target.
"""

from __future__ import annotations

import math

import mpmath as mp

from zeta_spectral_gpu import acceleration as acc


def _geometric_partial_sums(r: float, n: int) -> list[float]:
    """S_k = sum_{j=0}^{k} r^j  ->  1/(1-r); error ~ r^{k+1} (geometric)."""
    out, s, term = [], 0.0, 1.0
    for _ in range(n):
        s += term
        out.append(s)
        term *= r
    return out


def test_aitken_accelerates_geometric():
    r = 0.7
    seq = _geometric_partial_sums(r, 8)
    limit = 1.0 / (1.0 - r)
    raw_err = abs(seq[-1] - limit)
    acc_seq = acc.aitken(seq)
    # One Aitken pass removes the single geometric component -> essentially exact.
    assert abs(acc_seq[-1] - limit) < 1e-12
    assert abs(acc_seq[-1] - limit) < raw_err


def test_shanks_multiple_passes():
    # Two geometric components: 0.8^n and 0.5^n. Iterated Aitken improves with
    # each pass (but is NOT the exact 2nd-order Shanks -- that is Wynn-epsilon,
    # checked separately below to nail this same sequence).
    n = 12
    seq = [2.0 + 0.8**k + 0.5**k for k in range(n)]
    one = acc.shanks(seq, passes=1)
    two = acc.shanks(seq, passes=2)
    assert abs(two[-1] - 2.0) < abs(one[-1] - 2.0)  # monotone improvement
    assert abs(two[-1] - 2.0) < 1e-2
    # The epsilon-algorithm (true Shanks e_2) handles both components at once.
    assert abs(acc.wynn_epsilon(seq) - 2.0) < 1e-9


def test_wynn_epsilon_leibniz_pi():
    # Leibniz partial sums for pi/4 converge like 1/n; Wynn-epsilon turns that
    # into rapid convergence.
    n = 20
    seq, s = [], 0.0
    for k in range(n):
        s += (-1) ** k / (2 * k + 1)
        seq.append(4.0 * s)
    raw_err = abs(seq[-1] - math.pi)
    est = acc.wynn_epsilon(seq)
    assert est is not None
    assert abs(est - math.pi) < 1e-9
    assert abs(est - math.pi) < raw_err


def test_neville_recovers_polynomial_limit():
    # f(h) = pi + 3 h - 2 h^2 + 0.5 h^3 sampled at a few h; Neville at h=0 = pi.
    def f(h):
        return math.pi + 3 * h - 2 * h * h + 0.5 * h**3

    nodes = [0.4, 0.3, 0.2, 0.1]
    values = [f(h) for h in nodes]
    est = acc.neville_extrapolate(nodes, values)
    assert abs(est - math.pi) < 1e-12


def test_richardson_inverse_log_constant():
    # The #65 use case: g(L) = c + a/L + b/L^2, recover c from samples in 1/L.
    c, a, b = 1.0, -0.7, 0.4
    Ls = [4.0, 6.0, 9.0, 13.0, 20.0]
    nodes = [1.0 / L for L in Ls]
    values = [c + a / L + b / L**2 for L in Ls]
    est = acc.richardson_limit(nodes, values)
    assert abs(est - c) < 1e-9
    # A trailing window (smallest nodes) still recovers c to good order.
    est_w = acc.richardson_limit(nodes, values, window=3)
    assert abs(est_w - c) < 1e-3


def test_mpmath_arithmetic_preserves_precision():
    # Geometric series in mpf at high dps -> accelerated estimate keeps the digits.
    mp.mp.dps = 60
    r = mp.mpf(7) / 10
    seq, s, term = [], mp.mpf(0), mp.mpf(1)
    for _ in range(10):
        s += term
        seq.append(s)
        term *= r
    limit = 1 / (1 - r)
    est = acc.wynn_epsilon(seq)
    assert isinstance(est, mp.mpf)
    assert abs(est - limit) < mp.mpf(10) ** (-40)


def test_degenerate_stage_returns_none_not_garbage():
    # A constant sequence has zero second differences everywhere -> aitken yields
    # all-None (no spurious finite value), and shanks falls back gracefully.
    flat = [3.0] * 5
    assert all(t is None for t in acc.aitken(flat))
    assert acc.shanks(flat, passes=1) == []
    assert acc.wynn_epsilon(flat) is None


def test_forward_tracks_input_limit_not_external_target():
    # Litmus: shift the whole input by a constant; the extrapolant shifts by the
    # same constant (it tracks the INPUT's limit, with no pull toward any fixed
    # external value). An inverse/fitting scheme could not have this property.
    base = _geometric_partial_sums(0.6, 8)
    shifted = [s + 5.0 for s in base]
    e0 = acc.wynn_epsilon(base)
    e1 = acc.wynn_epsilon(shifted)
    assert abs((e1 - e0) - 5.0) < 1e-9
