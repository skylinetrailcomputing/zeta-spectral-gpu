"""Dirichlet characters and L-function zeros for the prime-driven mirror model.

Sierra's massless-Dirac mirror model (issue #25, :mod:`dirac_mirror`) generalises
from the Riemann zeta function to any Dirichlet ``L``-function by replacing the
mirror reflection coefficients ``mu(n)/sqrt(n)`` with ``chi(n) mu(n)/sqrt(n)``
(arXiv:1404.4252 eq. 13.6 / 14.9, reviewed in arXiv:1601.01797 §XIII–XIV). Because
a Dirichlet character is completely multiplicative,

    1 / L(s, chi) = sum_{n>=1} chi(n) mu(n) / n^s ,

so the forward locator of :mod:`dirac_mirror` — the truncated partial sum of this
series — has ``|M'_z(n)|`` grow with the truncation exactly at the zeros of
``L(s, chi)``. The only new input over the zeta case is the character ``chi``,
which depends solely on its modulus: still pure number theory in, zeros out.

This module supplies the two forward-side pieces:

* the **Dirichlet characters** themselves (the new number-theoretic input), and
* an **independent** computation of the ``L``-function zeros (via mpmath) used
  only to *score* the located peaks — the ``L``-function analogue of
  :mod:`zeros`. Nothing here feeds a zero into the operator; the zeros are
  produced to compare against (forward, not inverse — see the README and the
  issue #25 ruling).

Scope: characters of **prime modulus** ``p`` — where ``(Z/pZ)*`` is cyclic, so a
single primitive root generates the whole character group — plus the odd
character mod 4 (the Dirichlet beta ``L``-function). General composite conductors
are a documented follow-up; prime moduli already exhibit both real (Legendre) and
genuinely complex characters, which is what the generalisation needs to show.
"""

from __future__ import annotations

import numpy as np

from .dirac_mirror import mobius_sieve


def _distinct_prime_factors(m: int) -> list[int]:
    """Distinct prime factors of ``m >= 1`` by trial division (small ``m`` only)."""
    factors: list[int] = []
    d = 2
    while d * d <= m:
        if m % d == 0:
            factors.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        factors.append(m)
    return factors


def is_prime(m: int) -> bool:
    return m >= 2 and _distinct_prime_factors(m) == [m]


def primitive_root(p: int) -> int:
    """Smallest primitive root modulo a prime ``p`` (a generator of ``(Z/pZ)*``).

    ``g`` is a generator iff ``g^{(p-1)/q} != 1 (mod p)`` for every prime ``q``
    dividing ``p-1``.
    """
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    if p == 2:
        return 1
    order = p - 1
    qs = _distinct_prime_factors(order)
    for g in range(2, p):
        if all(pow(g, order // q, p) != 1 for q in qs):
            return g
    raise ValueError(f"no primitive root found mod {p}")  # unreachable for prime p


def dirichlet_character(modulus: int, index: int = 1) -> np.ndarray:
    """Period-``modulus`` character table ``chi[0..modulus-1]`` (complex128).

    For a **prime** ``modulus = p`` the group ``(Z/pZ)*`` is cyclic of order
    ``p-1``; with a primitive root ``g`` the ``index``-th character is

        chi(g^m mod p) = exp(2*pi*i * index * m / (p-1)),    chi(0) = 0,

    taken mod ``p-1`` in ``index``. ``index = 0`` is the principal character
    (``chi = 1`` on the units, so ``L(s, chi_0) = zeta(s) * prod_{q|p}(1-q^-s)``
    shares zeta's zeros); ``index = (p-1)/2`` is the real quadratic (Legendre)
    character. ``modulus = 4`` is special-cased: ``index = 0`` principal,
    ``index = 1`` the odd character ``[0, 1, 0, -1]`` whose ``L`` is the Dirichlet
    beta function.

    The returned array is indexed by residue, so ``chi[n % modulus]`` is
    ``chi(n)``; it is the only number-theoretic input the ``L``-function locator
    consumes (forward, not inverse).
    """
    if modulus == 4:
        if index % 2 == 0:
            return np.array([0, 1, 0, 1], dtype=np.complex128)  # principal mod 4
        return np.array([0, 1, 0, -1], dtype=np.complex128)  # odd -> Dirichlet beta
    p = modulus
    if not is_prime(p):
        raise NotImplementedError(
            f"only prime moduli (and 4) are supported; {p} is composite"
        )
    order = p - 1
    j = index % order
    g = primitive_root(p)
    chi = np.zeros(p, dtype=np.complex128)
    val = 1
    for m in range(order):
        chi[val] = np.exp(2j * np.pi * j * m / order)
        val = (val * g) % p
    return chi


def is_real_character(char: np.ndarray, tol: float = 1e-12) -> bool:
    """True if every character value is real (``chi^2`` is principal)."""
    return bool(np.all(np.abs(np.asarray(char).imag) <= tol))


def lfunction_weights(
    char: np.ndarray, n: int, *, mu: np.ndarray | None = None
) -> np.ndarray:
    """Locator coefficients ``c(k) = chi(k) mu(k)`` for ``k = 1..n``.

    These are the Dirichlet-series coefficients of ``1 / L(s, chi)`` (eq. 13.6) —
    the reflection coefficients of the prime-driven mirror array with the
    character switched on. Feed the result to :func:`dirac_mirror.mobius_partial_sum`
    (or its GPU twin) via the ``weights`` argument. Returned as real ``float64``
    when the character is real (so the fast real locator path is used) and
    ``complex128`` otherwise.
    """
    char = np.asarray(char, dtype=np.complex128)
    q = char.size
    if mu is None:
        mu = mobius_sieve(n)
    k = np.arange(1, n + 1)
    w = char[k % q] * mu[1 : n + 1]
    return w.real.copy() if is_real_character(w) else w


def lfunction_value(s, char: np.ndarray, *, derivative: int = 0):
    """``L(s, chi)`` (or its ``s``-derivative) via mpmath — the comparison source.

    Thin wrapper over ``mpmath.dirichlet``, which sums ``sum chi(k) k^{-s}`` for a
    period-``q`` character supplied as ``[chi(0), ..., chi(q-1)]``. Independent of
    the locator; used to produce ground-truth zeros to score peaks against.
    """
    import mpmath as mp

    chi_list = [mp.mpc(complex(c)) for c in char]
    return mp.dirichlet(s, chi_list, derivative=derivative)


def lfunction_zeros(
    char: np.ndarray,
    *,
    e_max: float = 30.0,
    e_min: float = 1.0,
    scan_step: float = 0.05,
    dps: int = 25,
    im_tol: float = 1e-6,
) -> np.ndarray:
    """Ordinates ``E > 0`` with ``L(1/2 + iE, chi) = 0`` up to ``e_max`` (ground truth).

    Independent of the locator: scans ``|L|`` on the critical line for deep minima
    and refines each with mpmath ``findroot``. Under GRH the low zeros lie on the
    line; refined roots with ``|Im E| > im_tol`` (or that fail to be genuine
    zeros) are discarded. Returned to *score* the locator's peaks — never fed into
    it (the ``L``-function analogue of :func:`zeros.riemann_zero_ordinates`).
    """
    import mpmath as mp

    mp.mp.dps = dps
    chi_list = [mp.mpc(complex(c)) for c in char]

    def lc(z):
        return mp.dirichlet(mp.mpc("0.5", z), chi_list)

    grid = np.arange(e_min, e_max, scan_step)
    abs_l = np.array([float(abs(lc(mp.mpf(float(e))))) for e in grid])
    threshold = 0.3
    found: list[float] = []
    for i in range(1, grid.size - 1):
        if not (abs_l[i] < abs_l[i - 1] and abs_l[i] <= abs_l[i + 1]):
            continue
        if abs_l[i] >= threshold:
            continue
        try:
            root = mp.findroot(lc, mp.mpf(float(grid[i])))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(mp.im(root)) >= im_tol or abs(lc(mp.re(root))) >= 1e-8:
            continue
        e = float(mp.re(root))
        if e_min <= e <= e_max and (not found or abs(e - found[-1]) > 1e-3):
            found.append(e)
    return np.array(sorted(set(found)))
