"""Invariants for the forward Li-criterion probe (issue #52).

Small ``n_max`` / modest ``dps`` so they stay CI-friendly. The decisive checks
pin the forward computation three independent ways -- the exact closed form for
``lambda_1``, published constants for ``lambda_1..lambda_5``, and a zero-free
Cauchy-integral Taylor of ``log xi`` -- and assert the structural forward
guarantee (the result never reads a zero). The larger sweep lives in
``scripts/run_li_criterion.py``.
"""

from __future__ import annotations

import mpmath as mp

from zeta_spectral_gpu import li_criterion as li, plots

# Published Li coefficients (Keiper 1992; Coffey; OEIS A074760), enough digits to
# anchor an independent check.
PUBLISHED = {
    1: "0.0230957089661210331",
    2: "0.0923457352280466703",
    3: "0.2076389205543248037",
    4: "0.3687904794922416385",
    5: "0.5755427144611774524",
}


def test_lambda1_matches_closed_form():
    # lambda_1 = a_1 = 1 + gamma/2 - (1/2) log(4 pi), exact.
    mp.mp.dps = 50
    closed = 1 + mp.euler / 2 - mp.log(4 * mp.pi) / 2
    (lam1,) = li.li_coefficients(1, dps=50)
    assert abs(lam1 - closed) < mp.mpf(10) ** (-45)


def test_matches_published_values():
    coeffs = li.li_coefficients(5, dps=60)
    for n, ref in PUBLISHED.items():
        assert abs(coeffs[n - 1] - mp.mpf(ref)) < mp.mpf("1e-15")


def test_log_xi_constant_term():
    # a_0 = log xi(1) = log(1/2).
    a = li.log_xi_coefficients(4, dps=50)
    assert abs(a[0] - mp.log(mp.mpf(1) / 2)) < mp.mpf(10) ** (-45)
    # a_1 equals lambda_1 (the n=1 combination is just a_1).
    assert abs(a[1] - (1 + mp.euler / 2 - mp.log(4 * mp.pi) / 2)) < mp.mpf(10) ** (-45)


def test_agrees_with_independent_cauchy_taylor():
    # Independent, still-forward method: Taylor of log xi at s=1 by Cauchy
    # integral (no zeros), fed through the same lambda_n combination.
    n_max = 8
    with mp.workdps(60):

        def log_xi(s):
            return mp.log(
                mp.mpf(1)
                / 2
                * s
                * (s - 1)
                * mp.power(mp.pi, -s / 2)
                * mp.gamma(s / 2)
                * mp.zeta(s)
            )

        a = mp.taylor(log_xi, 1, n_max, method="quad", radius=mp.mpf("0.5"))
        indep = [
            n * mp.fsum(mp.binomial(n - 1, j) * a[n - j] for j in range(n))
            for n in range(1, n_max + 1)
        ]
    ours = li.li_coefficients(n_max, dps=60)
    assert max(abs(ours[i] - indep[i]) for i in range(n_max)) < mp.mpf(10) ** (-40)


def test_log_power_series_recurrence():
    # log(1/(1-u)) = -log(1-u) = sum_{k>=1} u^k / k, so g[k] = 1/k.
    with mp.workdps(40):
        f = [mp.mpf(1)] * 9  # 1/(1-u) has all coefficients 1
        g = li._log_power_series(f, 8)
        assert g[0] == 0
        for k in range(1, 9):
            assert abs(g[k] - mp.mpf(1) / k) < mp.mpf(10) ** (-35)


def test_all_positive_consistent_with_rh():
    # The forward verdict over a modest range: every lambda_n > 0 (RH-consistent),
    # and they are strictly increasing here (the low coefficients climb).
    coeffs = li.li_coefficients(20)
    assert all(c > 0 for c in coeffs)
    assert all(coeffs[i] < coeffs[i + 1] for i in range(len(coeffs) - 1))


def test_evaluate_reports_verdict_and_is_stable():
    res = li.evaluate(15)
    assert res.rh_consistent
    assert res.all_positive
    assert res.min_index == 1  # smallest coefficient is lambda_1 in this range
    assert res.min_value > 0
    assert res.stability < mp.mpf("1e-20")  # well-resolved, not cancellation noise


def test_main_term_tracks_growth():
    # lambda_n settles onto the RH asymptotic (n/2)(log n + gamma - 1 - log 2pi):
    # the relative deviation shrinks as n grows (it is large at small n, where the
    # leading term has the wrong sign, and small in the tail).
    res = li.evaluate(20)
    rel = res.main_term_relative_error()  # n = 2 .. 20, all positive
    assert all(r > 0 for r in rel)
    assert rel[-1] < rel[0]  # closer to the asymptotic at n=20 than n=2
    assert rel[-1] < mp.mpf("0.3")


def test_forward_no_zeros_consumed(monkeypatch):
    # Structural guarantee: the computation must not read a single zeta zero.
    # Poison mpmath.zetazero so any inverse-style access would explode.
    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("li_criterion must not consume zeta zeros (forward only)")

    monkeypatch.setattr(mp, "zetazero", _boom)
    coeffs = li.li_coefficients(6, dps=50)
    assert all(c > 0 for c in coeffs)


def test_figure_renders(tmp_path):
    res = li.evaluate(12)
    out = plots.li_coefficients_figure(res, out_path=tmp_path / "li.png")
    assert out.exists() and out.stat().st_size > 0
