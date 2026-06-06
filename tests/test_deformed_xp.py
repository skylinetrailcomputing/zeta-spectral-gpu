"""Invariants for Sierra's deformed-xp forward spectrum (#23).

The CPU secular reference is the "truth" spectrum: the real roots of eq. 14. These
checks confirm (a) the operator is self-adjoint (the secular function is real),
(b) every returned eigenvalue is an actual root, (c) the spectrum lives above the
classical energy floor and reproduces the *average* Riemann density — without ever
consuming a zero as input (forward, not inverse). They run on CPU at modest
precision so they stay CI-friendly.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import deformed_xp as dxp


def test_secular_is_real_and_matches_shortcut():
    # eq. 14 is real for real E (self-adjoint H), and the one-Bessel shortcut
    # 2 Re[e^{-i theta/2} K_{1/2+iE/2}] equals the full two-term form.
    mp.mp.dps = 30
    for E in (3.0, 14.0, 25.0, 40.0):
        full = dxp._secular_full(E)
        assert abs(float(full.imag)) < 1e-20
        assert abs(float(full.real) - float(dxp.secular(E))) < 1e-25


def test_returned_eigenvalues_are_secular_roots():
    # The decisive correctness check: each E_n actually solves Xi(E_n) = 0.
    spec = dxp.secular_spectrum(6, cache_dir=None)
    dps = dxp.required_dps(6)
    with mp.workdps(dps):
        for E in spec:
            assert abs(float(dxp.secular(E))) < 1e-12


def test_spectrum_above_classical_bound_and_ascending():
    # Classical energies obey |E| >= 2h = 4*pi (eq. 6); the quantum levels too.
    spec = dxp.secular_spectrum(8, cache_dir=None)
    assert spec.shape == (8,)
    assert np.all(spec > dxp.CLASSICAL_BOUND)
    assert np.all(np.diff(spec) > 0)


def test_reproduces_average_density():
    # Analytic unfolding by the smooth Riemann-von Mangoldt count gives unit mean
    # spacing -> the spectrum reproduces the *average* zero density. The spacings
    # tighten toward 1 with height (the asymptotic regime).
    spec = dxp.secular_spectrum(10, cache_dir=None)
    s = np.diff(dxp.analytic_unfold(spec))
    assert abs(s.mean() - 1.0) < 0.05
    assert abs(s[-1] - 1.0) < abs(s[0] - 1.0)  # converging to 1


def test_roots_approach_average_zeros():
    # Forward claim: the levels track the *average* zeros (built from the smooth
    # count only). The match is asymptotic — the gap is positive and shrinking.
    spec = dxp.secular_spectrum(8, cache_dir=None)
    gap = spec - dxp.average_zeros(8)
    assert np.all(gap > 0)
    assert np.all(np.diff(gap) < 0)  # monotonically closing


def test_self_adjoint_extension_shifts_the_spectrum():
    # theta parameterises the U(1) family of self-adjoint extensions; a different
    # theta is a genuinely different operator -> a shifted spectrum (confirms the
    # extension angle is wired through, and that nothing is fit to the zeros).
    a = dxp.secular_spectrum(5, theta=np.pi / 4, cache_dir=None)
    b = dxp.secular_spectrum(5, theta=np.pi, cache_dir=None)
    assert np.all(np.abs(a - b) > 1e-3)
