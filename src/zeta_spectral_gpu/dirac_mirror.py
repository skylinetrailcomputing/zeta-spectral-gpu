"""Sierra's prime-driven massless-Dirac model — a forward locator (issue #25).

Germán Sierra, *The Riemann zeros as energy levels of a Dirac fermion in a
potential built from the prime numbers in Rindler spacetime*, J. Phys. A **47**,
325204 (2014), arXiv:1404.4252; reviewed in arXiv:1601.01797 §X–XV.

A massless Dirac fermion in the right Rindler wedge (radial coordinate
``rho >= l_1 = 1``) is free except for delta-function "moving mirrors" at radial
positions ``l_n``, each a unitary 2x2 transfer matrix ``T_n`` with reflection
coefficient ``varrho_n`` (eq. 10.20). Amplitudes iterate from the boundary vector
``|A_1(vartheta)> = (1, e^{i vartheta})`` (eq. 10.19/10.22). The **prime-driven**
choice (eq. 11.4 / 13.5) places a mirror at every square-free integer,

    l_n = sqrt(n),     varrho_n = mu(n) / sqrt(n)        (mu = Moebius),

so the mirror for a prime ``p`` sits at ``sqrt(p)`` and the boundary->mirror
round trip has period ``log p`` — Berry's "primes as periodic orbits", made
concrete and energy-independent (the fermion is massless).

Forward, not inverse — and the distinction matters more here than anywhere else
in the repo (see ``_private/issue-25-forward-ruling.md`` and project-framing).
The model has two separable pieces with *opposite* forward status:

* **Piece A — the prime-built locator (FORWARD).** The decisive object is the
  finite, purely number-theoretic partial sum (eq. 12.20)

      M'_z(n) = sum_{k<=n} mu(k) k^{-z},   z = 1/2 + iE,   E real,

  whose magnitude *grows with the truncation n* exactly when ``E`` is a zero of
  zeta (Fig. 14), with the quantitative prediction ``|M'_z(n)| ~ log n / |Z'(E)|``
  at a simple zero (eq. 12.30). This consumes only ``mu(k)`` and ``k^{-z}`` —
  pure number theory — and the zeros come out. Same posture as the CCM flagship.
  ``mobius_partial_sum`` / ``growth_profile`` implement it; it is the forward
  deliverable.

* **Piece B — the per-zero ``vartheta`` tuning (the inverse trap).** Individual
  zeros become genuine normalizable eigenstates of the self-adjoint ``H_vartheta``
  only when the boundary phase is set to ``vartheta = -(theta(E) + (pi/2) sign
  Z'(E))`` (eq. 12.33), with ``theta`` the Riemann–Siegel theta and ``Z`` the
  Hardy function. Both are *explicit* functions of a real ``E``, but feeding a
  **known** zero in to set ``vartheta`` and then "recovering" it consumes the
  zero. ``normalizable_amplitude`` reproduces Sierra's Fig. 4 dichotomy
  (normalizable at a zero with tuned ``vartheta`` vs. continuum otherwise) purely
  to **validate the bound-state machinery** — it is *not* a zero-finder. The
  forward locator is Piece A.

Precision is the opposite regime from the flagship: ``|M'_z(n)|`` is ``O(1)``,
no catastrophic cancellation, so **fp64 is sufficient** (a test cross-checks
against mpmath). That is what makes the GPU version (a Moebius-weighted Dirichlet
partial sum over a dense ``E``-grid, plus the Dirichlet-L generalisation
``mu(n) -> chi(n) mu(n)``) a clean fp64 win, unlike the flagship's mpmath-bound
eigensolve. This module is the CPU reference; the GPU assembly is the follow-up.
"""

from __future__ import annotations

import numpy as np


def mobius_sieve(n_max: int) -> np.ndarray:
    """Möbius function ``mu(1..n_max)`` as an ``int8`` array (index 0 unused/0).

    Linear-ish sieve over smallest prime factors: ``mu(1)=1``; ``mu(n)=0`` if a
    prime square divides ``n``; else ``(-1)^{# distinct prime factors}``. The
    only number-theoretic input the model consumes — everything forward flows
    from this and the ``k^{-z}`` weights.
    """
    if n_max < 1:
        return np.zeros(1, dtype=np.int8)
    mu = np.ones(n_max + 1, dtype=np.int8)
    mu[0] = 0
    is_comp = np.zeros(n_max + 1, dtype=bool)
    primes: list[int] = []
    for i in range(2, n_max + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n_max:
                break
            is_comp[ip] = True
            if i % p == 0:
                mu[ip] = 0  # p^2 | ip
                break
            mu[ip] = -mu[i]
    return mu


def mobius_partial_sum(
    E: np.ndarray | float,
    n: int,
    *,
    sigma: float = 0.5,
    weights: np.ndarray | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``M'_z(n) = sum_{k<=n} c(k) k^{-(sigma+iE)}`` — the forward locator (Piece A).

    Vectorised over ``E`` (fp64 ``complex128``). ``c(k)`` defaults to ``mu(k)``
    (zeta); pass ``weights = chi(k) mu(k)`` for a Dirichlet ``L``-function
    (eq. 13.6 / 179). The series is the truncated Dirichlet series of
    ``1/zeta(sigma+iE)`` (eq. 12.11): ``|M'_z(n)|`` grows with ``n`` at a zero on
    the critical line ``sigma = 1/2`` and stays bounded / oscillates otherwise.

    No boundary phase ``vartheta`` and no zeros enter — this is the prime-only
    object the zeros are read *off* of.
    """
    E = np.asarray(E, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    if weights is None:
        if mu is None:
            mu = mobius_sieve(n)
        weights = mu[1 : n + 1].astype(np.float64)
    else:
        weights = np.asarray(weights[:n], dtype=np.float64)
    # k^{-(sigma+iE)} = k^{-sigma} * exp(-iE ln k); outer over (E, k).
    amp = weights * k**-sigma  # shape (n,)
    logk = np.log(k)  # shape (n,)
    phase = np.exp(-1j * np.multiply.outer(E, logk))  # (E.shape, n)
    return phase @ amp.astype(np.complex128)


def growth_profile(
    E: float,
    truncations: np.ndarray,
    *,
    sigma: float = 0.5,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``|M'_z(n)|`` at each truncation ``n`` in ``truncations`` (Fig. 14).

    Computes the cumulative partial sum once and samples it, so passing many
    truncations is cheap. The forward signature of a zero is a profile that
    *increases* with ``n``; off a zero it stays bounded.
    """
    truncations = np.asarray(truncations, dtype=np.int64)
    n_top = int(truncations.max())
    if mu is None:
        mu = mobius_sieve(n_top)
    k = np.arange(1, n_top + 1, dtype=np.float64)
    terms = (
        mu[1 : n_top + 1].astype(np.float64) * k**-sigma * np.exp(-1j * E * np.log(k))
    )
    cumulative = np.cumsum(terms)
    return np.abs(cumulative[truncations - 1])


def normalizable_amplitude(
    E: float,
    truncations: np.ndarray,
    *,
    eps: float = 0.25,
    vartheta: float | None = None,
    sigma: float = 0.5,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``<A_n|A_n>`` at each truncation ``n`` (Sierra's Fig. 4) — the bound-state bridge.

    The leading Magnus / ``eps -> 0`` closed form (eq. 12.12/12.14): writing
    ``M = M'_z(n)`` (``mobius_partial_sum``) and its phase ``Phi = -arg M``,

        <A_n|A_n> = 2 [ e^{-2 eps|M|} cos^2(d) + e^{+2 eps|M|} sin^2(d) ],
        d = (vartheta - Phi) / 2.

    This bridges Piece A (``|M'_z|`` growth) to the *physical* statement — a
    normalizable state. At a zero with ``vartheta`` tuned to the limiting phase
    (``riemann_tuning_phase(E)``, the default) the growing ``sin^2`` mode is
    killed and the norm density *decays* like ``e^{-2 eps|M|} = n^{-2 eps/|Z'(E)|}``
    (sum -> ``2 zeta(1 + 2 eps/|Z'(E)|)`` < inf, eq. 12.34); at a generic ``E`` the
    ``sin^2`` term survives and it stays ``O(1)`` (continuum).

    **Validation utility, not a forward zero-finder.** The default ``vartheta``
    uses ``theta(E)`` — Piece B of the forward ruling — to confirm the machinery
    reproduces the paper. Forward zero-location is Piece A
    (``mobius_partial_sum`` / ``growth_profile``). The full finite-``eps``
    transfer-matrix product (resonances) is a separate, later step.
    """
    if vartheta is None:
        vartheta = riemann_tuning_phase(E)
    truncations = np.asarray(truncations, dtype=np.int64)
    n_top = int(truncations.max())
    if mu is None:
        mu = mobius_sieve(n_top)
    k = np.arange(1, n_top + 1, dtype=np.float64)
    terms = (
        mu[1 : n_top + 1].astype(np.float64) * k**-sigma * np.exp(-1j * E * np.log(k))
    )
    M = np.cumsum(terms)[truncations - 1]
    absM = np.abs(M)
    d = 0.5 * (vartheta - (-np.angle(M)))
    return 2.0 * (
        np.exp(-2.0 * eps * absM) * np.cos(d) ** 2
        + np.exp(2.0 * eps * absM) * np.sin(d) ** 2
    )


# --- Explicit functions used for VALIDATION only (Piece B; never zero-lookup) ---
#
# theta(E) (Riemann–Siegel) and sign Z'(E) (Hardy) are elementary functions of a
# real argument, computable with no table of zeros. They appear here to (a) tune
# vartheta when reproducing Fig. 4 and (b) check the |M'_z(n)| ~ log n / |Z'(E)|
# growth-rate prediction (eq. 12.30). They must never enter a pipeline whose
# purpose is to produce zeros — that is the inverse trap the ruling forbids.


def riemann_siegel_theta(E: float) -> float:
    """Riemann–Siegel theta ``theta(E)`` via mpmath ``siegeltheta`` (``mpf``)."""
    import mpmath as mp

    return float(mp.siegeltheta(E))


def hardy_z_prime(E: float, *, h: float = 1e-3) -> float:
    """``Z'(E)`` of the Hardy function ``Z`` (central difference, mpmath ``siegelz``)."""
    import mpmath as mp

    return float((mp.siegelz(E + h) - mp.siegelz(E - h)) / (2 * h))


def riemann_tuning_phase(E: float) -> float:
    """``vartheta(E) = -(theta(E) + (pi/2) sign Z'(E))`` (eq. 12.33) — Piece B.

    The self-adjoint-extension phase that makes a zero ``E`` a normalizable state.
    Validation-only (see ``transfer_amplitudes``); not part of forward
    zero-finding.
    """
    return -(riemann_siegel_theta(E) + (np.pi / 2.0) * np.sign(hardy_z_prime(E)))
