"""Li's criterion as a forward, computable RH probe (issue #52).

Li's criterion: the Riemann Hypothesis holds **iff** every Li coefficient

    lambda_n = sum_rho [ 1 - (1 - 1/rho)^n ]            (n = 1, 2, 3, ...)

is non-negative, where ``rho`` runs over the nontrivial zeros (Li 1997). That
``sum_rho`` form **consumes the zeros** -- it is exactly the inverse formulation
this project forbids, and is *not* used here. Instead we compute the same numbers
the **forward** way, from the Taylor coefficients of ``log xi(s)`` at ``s = 1``
(Bombieri-Lagarias 1999):

    lambda_n = (1 / (n-1)!) d^n/ds^n [ s^(n-1) log xi(s) ]_{s=1}.

Writing ``s = 1 + u`` and ``log xi(1 + u) = sum_k a_k u^k`` this collapses to a
finite, zero-free combination of the ``a_k`` (see :func:`li_coefficients`):

    lambda_n = n * sum_{j=0}^{n-1} C(n-1, j) a_{n-j}.

The ``a_k`` come in closed form, one per factor of the completed zeta
``xi(s) = (1/2) s (s-1) pi^(-s/2) Gamma(s/2) zeta(s)``:

- ``log s``        -> ``(-1)^(k+1) / k``;
- ``-(s/2) log pi``-> ``-(log pi)/2`` at ``k = 1``, else ``0``;
- ``log Gamma(s/2)``-> ``2^(-k) psi^(k-1)(1/2) / k!`` (polygamma);
- ``log[(s-1) zeta(s)]`` -> the log of the entire series ``(s-1) zeta(s)``, whose
  coefficients are the **Stieltjes constants** ``gamma_m`` (the Laurent data of
  ``zeta`` at its pole), composed through a ``log(1 + series)`` recurrence.

So the only inputs are ``pi``, ``Gamma``/``psi`` at ``1/2``, and the Stieltjes
constants -- pure analytic data of ``xi``. **No zero is ever read.** The zeros
appear only as the yardstick: the computed ``lambda_n`` are *compared* to the
positivity RH predicts (and, in the tests, cross-checked against an independent
Cauchy-integral Taylor of ``log xi`` -- still zero-free). This makes Li positivity
a cheap **scalar** shadow of the same Weil positivity the flagship CCM operator
encodes as ``lambda_min(c) >= 0`` over a finite cutoff.

**Precision reality (the repo's recurring fp64 wall, in scalar form).** Forming
``lambda_n`` weights the ``a_k`` by binomials up to ``C(n-1, .) ~ 2^n`` while the
result is only ``O(n log n)``, so there is genuine cancellation; together with the
Stieltjes constants needing their own digits, the working precision must grow with
``n`` (empirically ``dps ~ n + 30`` keeps tens of digits; see
:func:`default_dps`). A plain fp64 sweep saturates after a few dozen coefficients
-- so this is an mpmath computation, not "one big float64 pass". The GPU charter
angle is therefore *not* a single deeper ``zeta`` Li sweep but the
**parallel-over-family** generalisation (GRH: ``lambda_n(chi) >= 0`` for a whole
family of Dirichlet characters), the Phase-2 follow-up -- see
``knowledge/li-criterion.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


def default_dps(n_max: int) -> int:
    """A safe working precision for ``lambda_1 .. lambda_{n_max}``.

    Calibrated against a two-precision agreement check (the highest-index, worst-
    cancellation coefficient still agreed to >40 digits at this setting for
    ``n_max`` up to 100): ``n_max + 30`` with a floor of 50. Only a starting
    guess; :func:`evaluate` records a ``stability`` residual so an under-resolved
    run is visible rather than silently wrong.
    """
    return max(50, n_max + 30)


def _log_power_series(coeffs: list, n_max: int) -> list:
    """Coefficients of ``log f`` for a power series ``f`` with ``f[0] = 1``.

    Standard ``g = log f`` recurrence from ``f' = g' f``: with ``g[0] = 0``,

        g[k] = f[k] - (1/k) sum_{i=1}^{k-1} i * g[i] * f[k-i].

    Used to take the logarithm of the entire series ``(s-1) zeta(s)`` at ``s = 1``
    (constant term ``1``) without ever leaving exact-coefficient arithmetic.
    """
    g = [mp.mpf(0)] * (n_max + 1)
    for k in range(1, n_max + 1):
        acc = mp.fsum(i * g[i] * coeffs[k - i] for i in range(1, k))
        g[k] = coeffs[k] - acc / k
    return g


def log_xi_coefficients(n_max: int, *, dps: int | None = None) -> list:
    """Taylor coefficients ``a_0 .. a_{n_max}`` of ``log xi(1 + u)`` (forward).

    ``a_k = [u^k] log xi(1 + u)``, assembled in closed form from the four factors
    of ``xi`` (see the module docstring). The constant ``a_0 = log xi(1) =
    -log 2`` is included for completeness; it never enters a Li coefficient. No
    zero of ``zeta`` is consumed -- the inputs are ``pi``, the polygamma values at
    ``1/2``, and the Stieltjes constants.
    """
    if dps is None:
        dps = default_dps(n_max)
    with mp.workdps(dps):
        # (s-1) zeta(s) = 1 + sum_{m>=1} (-1)^(m-1) gamma_{m-1}/(m-1)! u^m  (u=s-1):
        # the Laurent expansion zeta(s) = 1/(s-1) + sum_n (-1)^n gamma_n/n! (s-1)^n
        # multiplied by (s-1). Its log is the third factor's contribution.
        eta = [mp.mpf(1)]
        for m in range(1, n_max + 1):
            eta.append((-1) ** (m - 1) * mp.stieltjes(m - 1) / mp.factorial(m - 1))
        log_eta = _log_power_series(eta, n_max)

        a = [mp.mpf(0)] * (n_max + 1)
        a[0] = -mp.log(2)  # = log xi(1) = log(1/2)
        log_pi = mp.log(mp.pi)
        for k in range(1, n_max + 1):
            from_log_s = mp.mpf((-1) ** (k + 1)) / k
            from_log_pi = -log_pi / 2 if k == 1 else mp.mpf(0)
            from_log_gamma = mp.polygamma(k - 1, mp.mpf(1) / 2) / (
                mp.mpf(2) ** k * mp.factorial(k)
            )
            a[k] = from_log_s + log_eta[k] + from_log_pi + from_log_gamma
        return a


def li_coefficients(n_max: int, *, dps: int | None = None) -> list:
    """The Li coefficients ``lambda_1 .. lambda_{n_max}`` (forward, no zeros).

    From the ``log xi`` Taylor coefficients via the finite, zero-free combination

        lambda_n = n * sum_{j=0}^{n-1} C(n-1, j) a_{n-j}

    (the ``s^(n-1) log xi`` derivative, expanded around ``s = 1``). RH is
    equivalent to ``lambda_n >= 0`` for every ``n``; these values are produced to
    be *compared* against that prediction, never fitted.
    """
    if dps is None:
        dps = default_dps(n_max)
    a = log_xi_coefficients(n_max, dps=dps)
    with mp.workdps(dps):
        out = []
        for n in range(1, n_max + 1):
            s = mp.fsum(mp.binomial(n - 1, j) * a[n - j] for j in range(n))
            out.append(n * s)
        return out


def li_main_term(n, *, dps: int | None = None) -> mp.mpf:
    """RH asymptotic main term ``(n/2)(log n + gamma - 1 - log 2pi)``.

    Under RH ``lambda_n ~ (n/2)(log n - log 2pi + gamma - 1)`` for large ``n``,
    with an oscillatory ``O(sqrt n)`` remainder (Keiper, Coffey, Voros). Used only
    as the forward comparison guide for the growth law -- it does not feed the
    computation. Convergence is slow, so finite-``n`` ratios sit a little above
    this term.
    """
    if dps is not None:
        with mp.workdps(dps):
            n = mp.mpf(n)
            return (n / 2) * (mp.log(n) + mp.euler - 1 - mp.log(2 * mp.pi))
    n = mp.mpf(n)
    return (n / 2) * (mp.log(n) + mp.euler - 1 - mp.log(2 * mp.pi))


@dataclass
class LiCriterionResult:
    """Outcome of a forward Li-coefficient sweep over ``n = 1 .. n_max``."""

    n_max: int
    dps: int
    coefficients: list  # lambda_1 .. lambda_{n_max} (mpf)
    min_value: mp.mpf  # min_n lambda_n
    min_index: int  # 1-based n attaining the minimum
    all_positive: bool  # the RH-consistency verdict over the swept range
    stability: mp.mpf  # max relative disagreement vs a lower-precision recompute

    @property
    def rh_consistent(self) -> bool:
        """True if no negative Li coefficient was found in the swept range.

        A *negative* ``lambda_n`` would disprove RH; finding none is consistent
        with (but, being a finite range, does not prove) RH. The forward verdict.
        """
        return self.all_positive

    def main_term_relative_error(self) -> list:
        """``|lambda_n - main(n)| / lambda_n`` for ``n >= 2`` -- the growth tracker.

        The leading asymptotic ``main(n)`` is negative for small ``n`` (it changes
        sign near ``n ~ 9.6``), so a bare ratio is ill-behaved there; this relative
        deviation is positive throughout and shrinks toward 0 as ``n`` grows, the
        clean signal that ``lambda_n`` settles onto the RH growth law.
        """
        with mp.workdps(self.dps):
            return [
                abs(self.coefficients[n - 1] - li_main_term(n))
                / self.coefficients[n - 1]
                for n in range(2, self.n_max + 1)
            ]


def evaluate(n_max: int, *, dps: int | None = None) -> LiCriterionResult:
    """Forward Li sweep with a built-in precision self-check.

    Computes ``lambda_1 .. lambda_{n_max}`` at ``dps`` (default :func:`default_dps`)
    and re-computes at a lower precision to estimate the worst-case relative
    disagreement (``stability``); a large value flags an under-resolved sweep
    rather than reporting cancellation noise as a (possibly negative) signal.
    """
    if dps is None:
        dps = default_dps(n_max)
    coeffs = li_coefficients(n_max, dps=dps)
    check = li_coefficients(n_max, dps=max(40, dps - max(15, n_max // 4)))
    with mp.workdps(dps):
        stability = max(
            (
                abs((coeffs[i] - check[i]) / coeffs[i])
                for i in range(n_max)
                if coeffs[i]
            ),
            default=mp.mpf(0),
        )
        min_value = min(coeffs)
        min_index = coeffs.index(min_value) + 1
        all_positive = all(c > 0 for c in coeffs)
    return LiCriterionResult(
        n_max=n_max,
        dps=dps,
        coefficients=coeffs,
        min_value=min_value,
        min_index=min_index,
        all_positive=all_positive,
        stability=stability,
    )
