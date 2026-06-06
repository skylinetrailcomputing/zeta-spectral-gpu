"""Invariants for Sierra's prime-driven massless-Dirac model (#25).

The decisive test is ``test_forward_locator_grows_at_zeros``: it confirms the
*forward* mechanism — the purely Möbius/prime-built partial sum ``M'_z(n)`` grows
with the truncation ``n`` at the Riemann zeros and stays small elsewhere
(Sierra's Fig. 14) — without ever feeding a zero in. ``test_fp64_matches_mpmath``
backs the precision claim that this regime needs no extended precision (unlike
the flagship). The ``normalizable_amplitude`` tests reproduce Fig. 4 to validate
the bound-state bridge; the ``vartheta`` tuning there is Piece B of the forward
ruling (validation only — see ``dirac_mirror`` docstring).
"""

from __future__ import annotations

import numpy as np

from zeta_spectral_gpu import dirac_mirror as dm

# First two nontrivial zero ordinates (used as probe points, not as operator input).
ZERO_1 = 14.134725
ZERO_2 = 21.022040
NONZERO = 17.0  # safely between the first two zeros


def test_mobius_sieve_small_values():
    mu = dm.mobius_sieve(12)
    assert list(int(x) for x in mu[1:13]) == [1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
    # mu(p) = -1 for primes; mu(p^2 k) = 0.
    assert mu[1] == 1 and mu[0] == 0


def test_fp64_matches_mpmath():
    # The locator lives in an O(1) regime with no catastrophic cancellation, so
    # fp64 reproduces an mpmath partial sum to ~1e-12 (the GPU win, per the memo).
    import mpmath as mp

    def mp_partial(E, n, sigma=0.5):
        mu_ = dm.mobius_sieve(n)
        z = mp.mpc(sigma, E)
        return complex(sum(int(mu_[k]) * mp.power(k, -z) for k in range(1, n + 1)))

    for E, n in [(ZERO_1, 150), (NONZERO, 150), (30.0, 500)]:
        a = complex(dm.mobius_partial_sum(E, n))
        b = mp_partial(E, n)
        assert abs(a - b) < 1e-11


def test_forward_locator_grows_at_zeros():
    # Fig. 14: |M'_z(n)| increases with n at a zero and is large; at a non-zero E
    # it stays small. Purely prime-built — no zeros consumed.
    tr = np.array([50, 100, 150])
    g1 = dm.growth_profile(ZERO_1, tr)
    g2 = dm.growth_profile(ZERO_2, tr)
    gn = dm.growth_profile(NONZERO, tr)
    assert g1[2] > g1[0] and g2[2] > g2[0]  # grows with truncation at a zero
    assert g1[-1] > 3.0 and g2[-1] > 3.0  # large at a zero
    assert gn[-1] < 1.5  # small off a zero
    assert g1[-1] > 3.0 * gn[-1]  # the zero stands out sharply


def test_growth_rate_matches_prediction():
    # eq. 12.30: |M'_z(n)| ~ log n / |Z'(E)| at a simple zero. Check within 30%.
    n = 150
    predicted = np.log(n) / abs(dm.hardy_z_prime(ZERO_1))
    measured = dm.growth_profile(ZERO_1, np.array([n]))[0]
    assert abs(measured - predicted) / predicted < 0.30


def test_mobius_partial_sum_vectorized_over_E():
    E = np.array([ZERO_1, NONZERO, ZERO_2])
    out = dm.mobius_partial_sum(E, 150)
    assert out.shape == (3,)
    singles = np.array([complex(dm.mobius_partial_sum(float(e), 150)) for e in E])
    np.testing.assert_allclose(out, singles, atol=1e-12)


def test_normalizable_amplitude_decays_at_zero_not_continuum():
    # Fig. 4: at a zero with tuned vartheta the norm density decays (normalizable);
    # at a generic E it stays O(1) (continuum). Validation of the bound-state
    # bridge — the vartheta tuning is Piece B (not forward zero-finding).
    tr = np.arange(10, 2001)
    amp_zero = dm.normalizable_amplitude(ZERO_1, tr, eps=0.25)  # default tuned vartheta
    amp_cont = dm.normalizable_amplitude(24.0, tr, eps=0.5, vartheta=np.pi)
    assert amp_zero[-200:].mean() < 0.1  # decays toward 0
    assert amp_zero[-1] < amp_zero[0]  # monotone-ish decay
    assert amp_cont[-200:].mean() > 1.0  # does not decay (continuum)


def test_lfunction_weights_hook_runs():
    # Dirichlet-L generalisation (eq. 13.6): swap mu(k) -> chi(k) mu(k). Smoke test
    # that the weights hook accepts a custom coefficient array.
    n = 100
    mu = dm.mobius_sieve(n)
    chi = np.ones(n + 1)  # trivial character -> recovers zeta
    weights = chi[1 : n + 1] * mu[1 : n + 1]
    a = complex(dm.mobius_partial_sum(ZERO_1, n, weights=weights))
    b = complex(dm.mobius_partial_sum(ZERO_1, n, mu=mu))
    np.testing.assert_allclose(a, b, atol=1e-12)
