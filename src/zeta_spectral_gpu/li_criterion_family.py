"""Phase-2 of #52: the GRH Li criterion over a family of Dirichlet L-functions.

Forward generalisation of :mod:`li_criterion` (the single-``zeta`` sweep) to
Dirichlet ``L``-functions. The **Generalized** Riemann Hypothesis for ``L(s, chi)``
is equivalent to the non-negativity of *its* Li coefficients ``lambda_n(chi)``,
computed forward from the completed ``L``-function

    Lambda(s, chi) = (q/pi)^{(s+a)/2} Gamma((s+a)/2) L(s, chi),

with ``a = 0`` for an even character (``chi(-1) = +1``) and ``a = 1`` for an odd one.
``Lambda(., chi)`` is entire of order 1 (for primitive non-principal ``chi``) with
zeros exactly the nontrivial zeros of ``L(s, chi)``, so the same Bombieri-Lagarias
identity carries over: writing ``log Lambda(1 + u, chi) = sum_k a_k(chi) u^k``,

    lambda_n(chi) = n * sum_{j=0}^{n-1} C(n-1, j) a_{n-j}(chi)

(:func:`li_criterion.li_from_log_coefficients`) -- a finite, **zero-free**
combination. The only new number-theoretic input over the ``zeta`` case is the
character ``chi`` (its modulus): the ``a_k(chi)`` come from the parity-aware
``Gamma``/polygamma at ``(1+a)/2`` plus the Taylor coefficients of ``log L(1+u, chi)``
(the character's own generalized-Stieltjes data, here in closed form from the
generalized Stieltjes constants ``gamma_n(r/q)`` -- ``mpmath.stieltjes`` -- never a
zero finder).

**The verdict.** For a **complex** character ``lambda_n(chi)`` is complex, and GRH is
equivalent to ``Re lambda_n(chi) >= 0`` for all ``n`` (Omar-Mazhouda 2007). For a
**real** (quadratic) character the multiset of zeros is conjugate-symmetric, so
``lambda_n(chi)`` is already real -- the imaginary part comes out ``~0``, a built-in
sanity check (``imag_residual``).

**Why this is the GPU angle.** Each character is an independent, zero-free forward
computation, so a whole family of hundreds of characters is embarrassingly parallel
-- mirroring the Katz-Sarnak ``#51 -> batched-kernel #68`` split. This module is the
``mpmath`` **CPU reference** (the precision-delicate analytic inputs stay here, per
CLAUDE.md's precision rule); :mod:`li_criterion_family_gpu` batches the fp64 family
*assembly* (the embarrassingly-parallel part), with the house GPU-vs-CPU agreement
test on small ``n``. A GPU generalized-Stieltjes kernel that would also produce the
analytic inputs on-device (for deeper ``n``) is the documented follow-up.

Forward, not inverse: characters in, Li coefficients out; GRH-positivity is the
prediction compared against. No zero of any ``L``-function is consumed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import mpmath as mp
import numpy as np

from . import li_criterion as li
from .dirichlet import dirichlet_character, is_prime, is_real_character
from .katz_sarnak import fundamental_discriminants, quadratic_character


def character_parity(char: np.ndarray) -> int:
    """``a in {0, 1}``: ``0`` if ``chi(-1) = +1`` (even), ``1`` if ``chi(-1) = -1`` (odd).

    Sets the shift in the completed ``L``-function's gamma factor ``Gamma((s+a)/2)``.
    ``chi(-1)`` is ``+-1`` for every Dirichlet character; read it off the period array.
    """
    char = np.asarray(char)
    return 0 if char[-1].real > 0 else 1


@functools.lru_cache(maxsize=None)
def _stieltjes_row(q: int, r: int, n_max: int, dps: int) -> tuple:
    """Generalized Stieltjes constants ``gamma_0 .. gamma_{n_max}`` at ``a = r/q``.

    Cached on ``(q, r, n_max, dps)`` so all characters of the same modulus reuse one
    Hurwitz-``zeta`` Laurent table (the per-modulus speed-up the family run relies on).
    """
    with mp.workdps(dps):
        a = mp.mpf(r) / q
        return tuple(mp.stieltjes(n, a) for n in range(n_max + 1))


def l_taylor_coefficients(char: np.ndarray, n_max: int, *, dps: int) -> list:
    """Taylor coefficients ``c_k = L^{(k)}(1, chi) / k!`` of ``L(1+u, chi)`` (forward).

    The character's **generalized-Stieltjes data**, in closed form. Writing the Hurwitz
    decomposition ``L(s, chi) = q^{-s} sum_{r=1}^{q-1} chi(r) zeta(s, r/q)`` and the
    Laurent series ``zeta(s, a) = 1/(s-1) + sum_n (-1)^n/n! gamma_n(a) (s-1)^n`` (the
    ``1/(s-1)`` poles cancel since ``sum_r chi(r) = 0`` for non-principal ``chi``):

        c_k = q^{-1} sum_{m+n=k} (-log q)^m/m! * (-1)^n/n! * G_n,
        G_n = sum_{r=1}^{q-1} chi(r) gamma_n(r/q),

    with ``gamma_n(a)`` the generalized Stieltjes constants (``mpmath.stieltjes``, cached
    per modulus in :func:`_stieltjes_row`). A forward evaluation -- no zero finder. For
    a primitive **non-principal** ``chi``, ``L`` is entire and ``c_0 = L(1, chi) != 0``
    (indeed ``c_0 = -q^{-1} sum_r chi(r) psi(r/q)``), so ``log L`` has an ordinary
    Taylor expansion. (``test_li_criterion_family`` anchors these against the
    independent Cauchy/Taylor route over the entire ``L``.)
    """
    char = np.asarray(char)
    q = char.size
    with mp.workdps(dps):
        G = [mp.mpf(0)] * (n_max + 1)
        for r in range(1, q):
            chi_r = mp.mpc(complex(char[r]))
            if chi_r == 0:
                continue
            row = _stieltjes_row(q, r, n_max, dps)
            for n in range(n_max + 1):
                G[n] += chi_r * row[n]
        log_q = mp.log(q)
        inv_q = mp.mpf(1) / q
        return [
            inv_q
            * mp.fsum(
                ((-log_q) ** m / mp.factorial(m))
                * (mp.mpf(-1) ** (k - m) / mp.factorial(k - m))
                * G[k - m]
                for m in range(k + 1)
            )
            for k in range(n_max + 1)
        ]


def completed_log_coefficients(char: np.ndarray, n_max: int, *, dps: int) -> list:
    """Taylor coefficients ``a_0 .. a_{n_max}`` of ``log Lambda(1+u, chi)`` (forward).

    Assembled in closed form from the three factors of the completed ``L``-function
    (see the module docstring), with ``arg = (1+a)/2`` the gamma argument:

    - ``(q/pi)^{(s+a)/2}``  -> ``arg*log(q/pi)`` at ``k=0``, ``(log(q/pi))/2`` at ``k=1``;
    - ``Gamma((s+a)/2)``    -> ``log Gamma(arg)`` at ``k=0``, ``2^{-k} psi^{(k-1)}(arg)/k!``;
    - ``log L(1+u, chi)``   -> ``log`` of the entire series ``L(1+u)/L(1)`` via the
      ``log(1 + series)`` recurrence (:func:`li_criterion._log_power_series`).

    ``a_0`` is included for completeness; it never enters a ``lambda_n``. No zero is
    consumed -- the inputs are ``q``, ``pi``, polygamma at ``arg``, and the ``L``-Taylor
    coefficients.
    """
    char = np.asarray(char)
    q = char.size
    a_par = character_parity(char)
    with mp.workdps(dps):
        c = l_taylor_coefficients(char, n_max, dps=dps)
        c0 = c[0]
        # log L(1+u) = log c0 + log(1 + sum_{k>=1} (c_k/c0) u^k); recurrence wants f[0]=1.
        norm = [mp.mpf(1)] + [c[k] / c0 for k in range(1, n_max + 1)]
        log_l = li._log_power_series(norm, n_max)  # log_l[0] = 0

        log_qpi = mp.log(mp.mpf(q) / mp.pi)
        half = mp.mpf(1) / 2
        arg = (1 + a_par) * half  # (1+a)/2 in {1/2 (even), 1 (odd)}

        a = [mp.mpf(0)] * (n_max + 1)
        a[0] = arg * log_qpi + mp.loggamma(arg) + mp.log(c0)
        for k in range(1, n_max + 1):
            from_qpi = half * log_qpi if k == 1 else mp.mpf(0)
            from_gamma = mp.polygamma(k - 1, arg) / (mp.mpf(2) ** k * mp.factorial(k))
            a[k] = log_l[k] + from_qpi + from_gamma
        return a


def character_li_coefficients(char: np.ndarray, n_max: int, *, dps: int | None = None):
    """The Li coefficients ``lambda_1 .. lambda_{n_max}`` of ``L(s, chi)`` (forward).

    From the ``log Lambda(., chi)`` Taylor coefficients via the same finite, zero-free
    combination as the ``zeta`` case (:func:`li_criterion.li_from_log_coefficients`).
    Returned as ``mpmath`` complex values; for a real character the imaginary parts
    are ``~0``.
    """
    if dps is None:
        dps = li.default_dps(n_max)
    a = completed_log_coefficients(char, n_max, dps=dps)
    with mp.workdps(dps):
        return li.li_from_log_coefficients(a, n_max)


@dataclass
class CharacterLiResult:
    """Outcome of a forward Li sweep for one Dirichlet character ``chi``."""

    label: str
    modulus: int
    parity: int  # 0 even, 1 odd
    is_real: bool
    dps: int
    coefficients: list  # lambda_1 .. lambda_{n_max} (mpc)
    min_re: mp.mpf  # min_n Re lambda_n -- the GRH-positivity margin
    min_re_index: int  # 1-based n attaining it
    all_re_positive: bool  # the per-character GRH-consistency verdict
    imag_residual: mp.mpf  # max_n |Im lambda_n| (~0 for real chi: a sanity check)
    stability: mp.mpf  # worst relative disagreement vs a lower-precision recompute

    @property
    def n_max(self) -> int:
        return len(self.coefficients)

    @property
    def rh_consistent(self) -> bool:
        """True if no ``Re lambda_n`` is negative over the swept range (GRH-consistent)."""
        return self.all_re_positive


def evaluate_character(
    char: np.ndarray, n_max: int, *, dps: int | None = None, label: str | None = None
) -> CharacterLiResult:
    """Forward per-character Li sweep with a built-in precision self-check.

    Computes ``lambda_1 .. lambda_{n_max}`` at ``dps`` and recomputes at a lower
    precision to estimate the worst relative disagreement (``stability``), so an
    under-resolved sweep is visible rather than reported as a (possibly negative)
    signal -- the scalar fp64-wall guard, mirroring :func:`li_criterion.evaluate`.
    The verdict is taken on ``Re lambda_n`` (real for a real character; the imaginary
    residual is recorded as a cross-check).
    """
    if dps is None:
        dps = li.default_dps(n_max)
    coeffs = character_li_coefficients(char, n_max, dps=dps)
    check = character_li_coefficients(
        char, n_max, dps=max(40, dps - max(15, n_max // 4))
    )
    char_arr = np.asarray(char)
    with mp.workdps(dps):
        re = [mp.re(c) for c in coeffs]
        stability = max(
            (
                abs((coeffs[i] - check[i]) / coeffs[i])
                for i in range(n_max)
                if coeffs[i]
            ),
            default=mp.mpf(0),
        )
        imag_residual = max((abs(mp.im(c)) for c in coeffs), default=mp.mpf(0))
        min_re = min(re)
        min_re_index = re.index(min_re) + 1
        all_re_positive = all(r > 0 for r in re)
    return CharacterLiResult(
        label=label or f"chi mod {char_arr.size}",
        modulus=int(char_arr.size),
        parity=character_parity(char_arr),
        is_real=bool(is_real_character(char_arr)),
        dps=dps,
        coefficients=coeffs,
        min_re=min_re,
        min_re_index=min_re_index,
        all_re_positive=all_re_positive,
        imag_residual=imag_residual,
        stability=stability,
    )


@dataclass
class FamilyLiResult:
    """Outcome of a forward Li sweep over a whole Dirichlet family."""

    kind: str
    n_max: int
    dps: int
    members: list = field(default_factory=list)  # CharacterLiResult

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def all_positive(self) -> bool:
        """GRH-consistency across the entire family (every ``Re lambda_n(chi) >= 0``)."""
        return all(m.all_re_positive for m in self.members)

    @property
    def rh_consistent(self) -> bool:
        return self.all_positive

    @property
    def worst_member(self) -> CharacterLiResult:
        """The member with the smallest ``min_re`` -- the tightest GRH margin (forward)."""
        return min(self.members, key=lambda m: m.min_re)

    @property
    def max_imag_residual(self) -> mp.mpf:
        """Largest ``|Im lambda_n|`` over all *real* members (should stay ``~0``)."""
        reals = [m.imag_residual for m in self.members if m.is_real]
        return max(reals, default=mp.mpf(0))


def evaluate_family(
    characters: list[tuple[str, np.ndarray]],
    n_max: int,
    *,
    dps: int | None = None,
    kind: str = "custom",
) -> FamilyLiResult:
    """Forward Li sweep over a list of ``(label, character)`` pairs.

    Each member is an independent, zero-free computation (:func:`evaluate_character`);
    the family-level GRH verdict is ``all_positive`` over every member. ``kind`` is a
    label for the family (e.g. ``"quadratic"`` / ``"prime"``).
    """
    if dps is None:
        dps = li.default_dps(n_max)
    members = [
        evaluate_character(char, n_max, dps=dps, label=label)
        for label, char in characters
    ]
    return FamilyLiResult(kind=kind, n_max=n_max, dps=dps, members=members)


# --- Family enumerators (the number-theoretic input) --------------------------


def _primes_upto(n: int) -> list[int]:
    return [p for p in range(3, n + 1) if is_prime(p)]


def quadratic_family(
    q_max: int, *, real: bool = True, imaginary: bool = True
) -> list[tuple[str, np.ndarray]]:
    """Real (quadratic) characters of fundamental discriminant ``0 < |d| <= q_max``.

    Reuses :func:`katz_sarnak.fundamental_discriminants` /
    :func:`katz_sarnak.quadratic_character`: each ``chi_d(n) = (d | n)`` is a real
    primitive character of conductor ``|d|`` (the symplectic Katz-Sarnak family). All
    real -> ``lambda_n`` real.
    """
    return [
        (f"chi_{d}", quadratic_character(d))
        for d in fundamental_discriminants(q_max, real=real, imaginary=imaginary)
    ]


def prime_family(q_max: int) -> list[tuple[str, np.ndarray]]:
    """All non-principal characters of prime modulus ``p <= q_max``.

    For prime ``p`` the group ``(Z/pZ)*`` is cyclic of order ``p-1``; the characters
    ``chi_{p,j}`` for ``j = 1 .. p-2`` are exactly the primitive non-principal ones
    (the principal ``j = 0`` -- whose ``L`` has a pole at ``s = 1`` -- is excluded).
    Includes genuinely **complex** characters (``j != (p-1)/2``), exercising the
    ``Re lambda_n`` verdict.
    """
    out: list[tuple[str, np.ndarray]] = []
    for p in _primes_upto(q_max):
        for j in range(1, p - 1):
            out.append((f"chi_{p}.{j}", dirichlet_character(p, j)))
    return out


def dirichlet_family(kind: str, q_max: int) -> list[tuple[str, np.ndarray]]:
    """Enumerate a named Dirichlet family up to conductor/modulus ``q_max``.

    ``kind``: ``"quadratic"`` (real characters, symplectic family), ``"prime"`` (all
    non-principal characters of prime modulus, real and complex), or ``"all"`` (both).
    """
    if kind == "quadratic":
        return quadratic_family(q_max)
    if kind == "prime":
        return prime_family(q_max)
    if kind == "all":
        return quadratic_family(q_max) + prime_family(q_max)
    raise ValueError(f"unknown family kind {kind!r} (quadratic / prime / all)")
