"""Davenport-Heilbronn negative control (#85).

The decisive checks are the control readouts themselves: a *genuine* off-line
zero is computed as output and verified at high precision (the upgrade of the
#43 planted-counterfactual demo), the growth-law discriminator fires on it, and
the on-line zero harvest shows the off-line deficit against the smooth count.
The fast tests pin the constructions (kappa from the Gauss sum, the functional
equation, the fp64 Euler-Maclaurin evaluator, Dirichlet-series inversion); the
``slow`` test runs the larger statistics harvest.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from zeta_spectral_gpu import davenport_heilbronn as dh
from zeta_spectral_gpu import spacing
from zeta_spectral_gpu.dirichlet import dirichlet_character, lfunction_value

# Balanzario & Sanchez-Ortiz (2007) tabulate the first off-line zero near
# 0.808517 + 85.699348i; used ONLY to validate our independently computed root.
FIRST_OFF_LINE = 0.808517 + 85.699348j


# --- Construction: kappa, coefficients, functional equation -------------------


def test_kappa_matches_classical_closed_form():
    # tan(arg(eps)/2) from the Gauss sum == (sqrt(10 - 2 sqrt 5) - 2)/(sqrt 5 - 1).
    closed = (np.sqrt(10.0 - 2.0 * np.sqrt(5.0)) - 2.0) / (np.sqrt(5.0) - 1.0)
    assert abs(dh.dh_kappa() - closed) < 1e-14


def test_coefficients_are_real_period_5():
    k = dh.dh_kappa()
    b = dh.dh_coefficients(12)
    expected = [1.0, k, -k, -1.0, 0.0]
    for n in range(1, 13):
        assert b[n] == pytest.approx(expected[(n - 1) % 5], abs=1e-15)


def test_dh_value_matches_component_combination():
    # f == ((1 - i kappa) L_chi + (1 + i kappa) L_chibar) / 2 — the two builds
    # of the same function agree (mpmath, one point is enough).
    k = dh.dh_kappa()
    chi = dirichlet_character(5, 1)
    s = mp.mpc("0.5", "10.0")
    combo = (
        (1 - 1j * k) * complex(lfunction_value(s, chi))
        + (1 + 1j * k) * complex(lfunction_value(s, np.conj(chi)))
    ) / 2.0
    assert abs(complex(dh.dh_value(s)) - combo) < 1e-15


def test_functional_equation_lambda_self_dual():
    # Lambda(s) = Lambda(1 - s) — the property that *defines* kappa. A wrong
    # sign convention anywhere (Gauss sum, parity, rotation) breaks this. The
    # residual floors at kappa's fp64 rounding (~1e-16 relative), not at dps.
    for s in (mp.mpc("0.7", "3.0"), mp.mpc("0.3", "11.5")):
        lhs = dh.dh_completed(s)
        rhs = dh.dh_completed(1 - s)
        assert float(abs(lhs - rhs) / abs(lhs)) < 1e-14


# --- fp64 evaluator vs mpmath --------------------------------------------------


def test_hurwitz_em_matches_mpmath():
    mp.mp.dps = 30
    for t, a in ((50.0, 0.2), (300.0, 0.8)):
        s = np.array([0.5 + 1j * t, 0.9 + 1j * t, 1.4 + 1j * t])
        ours = dh.hurwitz_zeta_em(s, a)
        for o, x in zip(ours, s):
            ref = complex(mp.zeta(mp.mpc(x), a))
            assert abs(o - ref) / abs(ref) < 1e-11


def test_line_scan_rotation_is_real_and_matches_mpmath():
    scan = dh.line_scan(np.arange(10.0, 100.0, 0.7))
    assert scan.residual < 1e-10  # all three Z's genuinely real after rotation
    t0 = 47.3
    z_ref = complex(dh.dh_value(mp.mpc("0.5", t0))) * np.exp(1j * dh.dh_theta(t0))
    assert abs(dh.line_scan(np.array([t0])).z_f[0] - z_ref.real) < 1e-10


# --- The control readouts -------------------------------------------------------


def test_on_line_zeros_are_zeros_and_track_smooth_count():
    found = dh.critical_line_zeros(2.0, 40.0)
    assert found.size >= 15
    mp.mp.dps = 25
    for t in found[::7]:
        assert float(abs(dh.dh_value(mp.mpc("0.5", float(t))))) < 1e-8
    predicted = float(dh.dh_smooth_count(40.0) - dh.dh_smooth_count(2.0))
    assert abs(found.size - predicted) < 3  # no off-line pair this low


def test_off_line_zero_census_finds_genuine_rh_violation():
    # FORWARD: characters + kappa in, the off-line zero out — then validated
    # against the published Balanzario & Sanchez-Ortiz location. This is the
    # genuine (not planted) counterexample object of the #43 upgrade.
    found = dh.off_line_zeros(80.0, 90.0)
    assert found.size == 1
    rho = complex(found[0])
    assert rho.real - 0.5 > 0.25  # well off the critical line
    assert abs(rho - FIRST_OFF_LINE) < 1e-4
    mp.mp.dps = 40
    assert float(abs(dh.dh_value(mp.mpc(rho.real, rho.imag), dps=40))) < 1e-12


def test_dirichlet_inverse_is_convolution_inverse():
    n = 400
    b = dh.dh_coefficients(n)
    c = dh.dirichlet_inverse(n)
    for m in range(1, n + 1):
        divisors = [d for d in range(1, m + 1) if m % d == 0]
        conv = sum(b[d] * c[m // d] for d in divisors)
        assert conv == pytest.approx(1.0 if m == 1 else 0.0, abs=1e-9)


def test_growth_dichotomy_fires_on_genuine_offline_zero():
    # The #43 growth law on a real off-line zero: slope ~ sigma_c - 1/2 at the
    # off-line zero, well above the (nonzero, no-Euler-product) background.
    c = dh.dirichlet_inverse(200_000)
    rho = complex(dh.off_line_zeros(84.0, 87.0)[0])
    off_slope = dh.growth_exponent(rho.imag, c, n_max=200_000)
    on = dh.critical_line_zeros(49.0, 51.0)[0]
    on_slope = dh.growth_exponent(float(on), c, n_max=200_000)
    assert abs(off_slope - (rho.real - 0.5)) < 0.08
    assert off_slope - on_slope > 0.15
    assert on_slope < 0.18  # elevated vs Moebius (~0) but far below the zero


@pytest.mark.slow
def test_statistics_control_repulsion_and_deficit():
    # The measured headline (knowledge note): f's on-line zeros KEEP GUE-level
    # repulsion at modest height (local statistics cannot tell f from a genuine
    # L-function); what betrays f is the zero-count deficit — off-line pairs
    # stolen from the line — plus the off-line census itself.
    z_f = dh.critical_line_zeros(5.0, 500.0)
    z_chi = dh.critical_line_zeros(5.0, 500.0, which="chi")
    deficit = float(dh.dh_smooth_count(500.0) - dh.dh_smooth_count(5.0)) - z_f.size
    assert deficit > 10.0  # ~19.5 measured; off-line pairs leave the line
    chi_deficit = (
        float(dh.dh_smooth_count(500.0) - dh.dh_smooth_count(5.0)) - z_chi.size
    )
    assert abs(chi_deficit) < 3.0  # the genuine L-function has no deficit

    r_f = float(np.mean(spacing.spacing_ratios(z_f)))
    union = np.sort(
        np.concatenate([z_chi, dh.critical_line_zeros(5.0, 500.0, which="chibar")])
    )
    r_union = float(np.mean(spacing.spacing_ratios(union)))
    assert abs(r_f - dh.MEAN_RATIO_GUE) < 0.06  # repulsion retained
    assert r_f - r_union > 0.1  # nothing like the 2-process superposition
