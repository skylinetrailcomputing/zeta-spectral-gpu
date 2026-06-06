"""Invariants for the Berry-Keating x<->p-symmetric deformed-xp count (#59).

H_II = (x + l_x^2/x)(p + l_p^2/p) has no closed-form secular equation (its metric
is curved, unlike H_I), so the forward, exactly-computable object is the
semiclassical Bohr-Sommerfeld counting function. These checks confirm (a) the
classical energy floor 4h, (b) monotonicity, (c) that the numerically integrated
area count converges to the derived leading asymptotic (E/2pi)(log(E/h) - 1) --
the decisive check that pins the scale (the same h as H_I, Sierra eq. 5.18),
independent of the quadrature, (d) the model's l_x l_p-scaling symmetry, and (e)
that at this same scale the count reproduces the average zeros' two leading terms
(up to the semiclassically-unpinned 7/8). Forward: no zero is ever consumed (the
comparison target is the smooth Riemann-von Mangoldt count).
"""

from __future__ import annotations

import numpy as np

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import deformed_xp_symmetric as sym


def test_floor_and_below():
    # No levels at or below the classical floor 4h (the fixed point energy); the
    # count turns on just above it. Contrast H_I, whose floor is 2h.
    h = sym.H_PRODUCT
    floor = sym.classical_bound(h)
    assert floor == 4.0 * h
    assert sym.classical_count(floor) == 0.0
    assert sym.classical_count(0.5 * floor) == 0.0
    assert sym.classical_count(1.5 * floor) > 0.0


def test_count_is_monotonic():
    es = np.linspace(1.1 * sym.classical_bound(), 5000.0, 12)
    counts = np.array([sym.classical_count(E) for E in es])
    assert np.all(np.diff(counts) > 0)


def test_leading_asymptotic_pins_the_h_scale():
    # Decisive derivation check. A(E)/E -> log(E/h) - 1, so
    #   r(E) = classical_count(E) * 2pi / E - log(E/h)  ->  -1,
    # at the *same* scale h as H_I (Sierra eq. 5.18). A wrong scale (e.g. 2h) would
    # make r converge to -1 + log 2 ~= -0.31 -- so this checks the area math against
    # the published asymptotic, independent of the quadrature.
    h = sym.H_PRODUCT

    def r(E):
        return sym.classical_count(E) * sym.TWO_PI / E - np.log(E / h)

    r_mid, r_far = r(2000.0), r(20000.0)
    assert abs(r_far + 1.0) < abs(r_mid + 1.0)  # converging to -1
    assert abs(r_far + 1.0) < 0.05


def test_count_matches_leading_term_at_height():
    # The closed-form leading_count is the leading term of the quadrature count;
    # they agree to better than 1% by mid height, improving with E.
    for E in (4000.0, 16000.0):
        rel = abs(sym.classical_count(E) - sym.leading_count(E)) / sym.classical_count(
            E
        )
        assert rel < 0.01


def test_lx_lp_scaling_symmetry():
    # Only the product h = l_x l_p enters: A(lambda E, lambda h) = lambda A(E, h).
    E, h, lam = 3000.0, sym.H_PRODUCT, 1.7
    a = sym.classical_area(E, h=h)
    a_scaled = sym.classical_area(lam * E, h=lam * h)
    np.testing.assert_allclose(a_scaled, lam * a, rtol=1e-9)


def test_matches_average_zeros_two_leading_terms():
    # The headline forward result: at the SAME scale as H_I (l_x l_p = 2pi, no
    # rescaling), H_II reproduces the average (smooth) zeros' two leading terms, so
    # n_II(E) - N_bar(E) converges to a constant -- the semiclassically-unpinned O(1)
    # offset, here -7/8 (no-Maslov convention) -- rather than growing with E.
    def diff(E):
        return sym.classical_count(E) - float(sym.average_count(E))

    d_mid, d_far = diff(4000.0), diff(40000.0)
    assert abs(d_far + 0.875) < abs(d_mid + 0.875)  # converging to -7/8
    assert abs(d_far + 0.875) < 0.05


def test_floor_is_twice_the_asymmetric_floor():
    # Concrete spectral difference from H_I: same scale h, but the symmetric
    # deformation lifts the classical floor from 2h (deformed_xp.CLASSICAL_BOUND)
    # to 4h -- the fixed point (l_x, l_p) sits higher than H_I's wall.
    np.testing.assert_allclose(
        sym.classical_bound(dxp.BESSEL_ARG), 2.0 * dxp.CLASSICAL_BOUND, rtol=1e-12
    )
