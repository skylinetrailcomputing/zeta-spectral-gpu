"""Convergence law of the CCM spectrum toward the zeta zeros (issue #65).

Forward. The spectrum comes from :mod:`ccm` — a finite-cutoff operator built from
the *primes* (a von Mangoldt sum cut at ``k <= lambda^2``); the zeros enter only
as the yardstick the spectrum is measured against. This module adds the
*convergence-law* layer on top, implementing Sliwinski (*Spectral Analysis of the
D_log^(lambda,N) Operators*, arXiv:2601.12133):

- the **mean absolute error** ``eps(lambda,N) = (1/n) sum_{k<=n} |nu_k - zeta_k|``
  and the **uniform error** ``E(lambda,N) = max_k |nu_k - zeta_k|`` (Defs 2.5/2.6);
- the proven **Heisenberg lower bound** ``eps >= 1/(4 ln lambda)`` (Thm 3.1) — a
  consequence of the operator living in a log-window of width ``2 ln lambda``;
- tools to extrapolate the cutoff-sequence of an eigenvalue toward the
  ``cutoff -> infinity`` limit (:func:`accelerate_zero`, via :mod:`acceleration`).

**The precision reality this module exists to make explicit (Phase-0 of #65).**
The spectrum needs the near-null eigenvector ``xi`` of the Weil form, which lives
at the sub-``eps_N`` scale; resolving it needs precision that *grows with the
cutoff depth* (fp64 walls at ``x ~ 5-9``; ``dps = 110`` at ``x ~ 30-50``; Groskin
reaches ``x = 100`` only at ``dps = 500-1000``). Two consequences, both load-bearing
for reading Sliwinski's numerics:

1. The genuine *low-zero* error is **super-exponential** in the cutoff (the §6
   table: ``3.4e-50 -> 2.4e-55 -> 1.1e-60`` at ``x = 12,13,14``), NOT inverse-log.
   The inverse-log behaviour is strictly the *aggregate / resolution-edge*
   statement (Thm 3.1), where the ``n``-th eigenvalue always sits at the window
   edge ``d_n = 2 pi n / L``.
2. A fp64 (``~7``-digit) spectrum is *corrupted*, not merely imprecise — its
   error over a fixed low set stays ``O(0.01..40)`` and cannot collapse. So a
   low-precision inverse-log "measurement" can be the precision wall in disguise.
   :func:`fp64_spectrum_corruption` quantifies exactly this.

Every routine carries a **resolution gate**: if the recovered first-zero error is
not small, ``xi`` was under-resolved at the requested ``dps`` and the result is
flagged, never reported as signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

from . import acceleration, ccm


# ----------------------------------------------------------------------------
# The error notions and the Heisenberg bound
# ----------------------------------------------------------------------------


def heisenberg_bound(lam) -> mp.mpf:
    """Sliwinski Thm 3.1 lower bound ``1/(4 ln lambda)`` on the mean abs error.

    The operator's log-window has width ``2 ln lambda``, capping the position
    spread; the uncertainty principle then bounds the spectral spread (hence the
    average eigenvalue-vs-zero distance) below by ``1/(4 ln lambda)``. In the
    repo's cutoff ``x = lambda^2`` this is ``1/(2 ln x)``.
    """
    return 1 / (4 * mp.log(lam))


def suggest_dps(x) -> int:
    """A safe ``dps`` heuristic for resolving ``xi`` at prime cutoff ``x``.

    Calibrated to the §6 / Groskin depths (first-zero error ``~1e-50`` at ``x=12``,
    losing ``~5`` digits per unit ``x``; Groskin's ``dps~500`` at ``x=100``). The
    minimal eigenvalue sits a little deeper than the zero error, so we add headroom
    and a floor. This is only a *starting* guess — the resolution gate in
    :func:`convergence_errors` is the actual safety net.
    """
    depth = 50.0 + 5.2 * max(0.0, float(x) - 12.0)
    return int(min(2000, max(60, 1.6 * depth)))


@dataclass
class ConvergenceErrors:
    """The two error notions at one ``(N, lambda)`` cell, plus diagnostics."""

    N: int
    lam: mp.mpf
    cutoff: mp.mpf  # x = lambda^2 (the prime cutoff)
    count: int  # zeros actually compared (<= N, capped by resolution)
    mae: mp.mpf  # eps(lambda,N): mean_{k<=count} |nu_k - zeta_k|
    uniform: mp.mpf  # E(lambda,N): max_k |nu_k - zeta_k|
    bound: mp.mpf  # 1/(4 ln lambda)
    first_zero_error: mp.mpf  # |nu_1 - zeta_1|, the resolution gate
    resolved: bool  # first_zero_error < resolve_tol
    errors: list  # per-index |nu_k - zeta_k|, k = 1..count
    eigenvalues: list  # nu_k
    zeros: list  # zeta_k

    @property
    def ln_lambda(self) -> mp.mpf:
        return mp.log(self.lam)

    @property
    def bound_ratio(self) -> mp.mpf:
        """Tracked-set MAE relative to the Heisenberg edge-floor ``1/(4 ln lambda)``.

        ``< 1`` means the *resolved* spectrum converges far better than the floor —
        which it does, super-exponentially (the floor is a property of the full
        first-``N`` set, dominated by the unresolvable resolution edge, not of the
        zeros the operator actually tracks). So this ratio being small is the
        finding, not a violation of Thm 3.1: the inverse-log floor is an edge
        phenomenon. See the module docstring.
        """
        return self.mae / self.bound


def convergence_errors(
    N: int,
    lam,
    count: int | None = None,
    *,
    dps: int | None = None,
    resolve_tol: float = 1e-3,
) -> ConvergenceErrors:
    """Mean-absolute and uniform error of the CCM spectrum vs the first zeros.

    ``count`` defaults to ``N`` — the full first-``N`` set Thm 3.1 is stated over,
    whose top eigenvalue always sits at the resolution edge (this is what forces
    ``eps`` above ``1/(4 ln lambda)``). Fewer eigenvalues than ``count`` may be
    returned if the operator's range runs out first; the actual number compared is
    recorded in ``count``. ``dps`` defaults to :func:`suggest_dps`. The result's
    ``resolved`` flag is ``False`` when the first-zero error exceeds ``resolve_tol``
    — the sign that ``xi`` was under-resolved (e.g. fp64, or ``dps`` too small for
    the cutoff depth); such a cell is returned but must not be read as signal.
    """
    lam = mp.mpf(lam)
    if count is None:
        count = N
    if dps is None:
        dps = suggest_dps(lam**2)

    res = ccm.converge(N, lam, count, dps=dps)
    with mp.workdps(dps):
        errors = list(res.errors)
        m = len(errors)
        mae = mp.fsum(errors) / m if m else mp.inf
        uniform = max(errors) if m else mp.inf
        bound = heisenberg_bound(lam)
        first = errors[0] if m else mp.inf
    return ConvergenceErrors(
        N=N,
        lam=lam,
        cutoff=lam**2,
        count=m,
        mae=mae,
        uniform=uniform,
        bound=bound,
        first_zero_error=first,
        resolved=bool(first < resolve_tol),
        errors=errors,
        eigenvalues=list(res.eigenvalues),
        zeros=list(res.zeros),
    )


# ----------------------------------------------------------------------------
# The precision-artifact quantifier (the headline Phase-0 finding)
# ----------------------------------------------------------------------------


@dataclass
class Fp64Corruption:
    """How far a fp64 spectrum sits from the true (mpmath) one at one cell."""

    N: int
    cutoff: mp.mpf
    eps_fp64: float  # the fp64 minimal |eigenvalue| (already roundoff if tiny)
    max_vs_mpmath: float  # max_k |nu_k^{fp64} - nu_k^{mpmath}| over the compared set
    max_vs_zeros_fp64: float  # max_k |nu_k^{fp64} - zeta_k| (the "error" fp64 reports)
    max_vs_zeros_mpmath: float  # max_k |nu_k^{mpmath} - zeta_k| (the genuine error)


def fp64_spectrum_corruption(
    N: int, lam, count: int, *, dps: int | None = None
) -> Fp64Corruption:
    """Quantify the fp64 spectrum's departure from the true spectrum.

    Recovers ``xi`` two ways — fp64 ``eigh`` of the fp64-assembled Weil matrix vs
    the mpmath inverse-iteration reference — feeds *both* through the same secular
    root-finder, and compares. The point (Phase-0 of #65): the fp64 spectrum's
    error against the zeros (``max_vs_zeros_fp64``) is dominated by ``xi``-corruption
    (``max_vs_mpmath``), not by genuine finite-cutoff error (``max_vs_zeros_mpmath``,
    which is super-exponentially smaller for the low zeros). So a fp64 inverse-log
    "measurement" is largely measuring the precision wall.
    """
    from . import ccm_gpu

    lam = mp.mpf(lam)
    if dps is None:
        dps = suggest_dps(lam**2)
    L = 2 * mp.log(lam)

    # Genuine spectrum (mpmath) and zeros.
    mp_spec = ccm.operator_spectrum(N, lam, count=count, dps=dps)
    with mp.workdps(dps):
        zeros = ccm.reference_ordinates(count)

    # fp64 xi -> secular roots (root-find kept in mpmath so only xi differs).
    A = ccm_gpu.assemble_weil_matrix_fp64(N, float(lam))
    w, V = np.linalg.eigh(A)
    i = int(np.argmin(np.abs(w)))
    v = V[:, i]
    even = np.array([(v[N + n] + v[N - n]) / 2.0 for n in range(-N, N + 1)])
    even = even / np.linalg.norm(even)
    with mp.workdps(50):
        fp64_roots = ccm.operator_eigenvalues(
            [mp.mpf(float(c)) for c in even], N, L, count
        )

    m = min(len(mp_spec), len(fp64_roots), count)
    # The genuine error is super-exponentially small (~1e-40..1e-60) while the
    # eigenvalues are O(100); subtracting in float would round it to 0 (below
    # float's RELATIVE precision at that magnitude). Keep it in mpmath.
    with mp.workdps(dps):
        genuine = max(abs(mp_spec[k] - zeros[k]) for k in range(m)) if m else mp.inf
    fp64_f = np.array([float(s) for s in fp64_roots[:m]])
    mp_f = np.array([float(s) for s in mp_spec[:m]])
    zeros_f = np.array([float(z) for z in zeros[:m]])
    return Fp64Corruption(
        N=N,
        cutoff=lam**2,
        eps_fp64=float(w[i]),
        max_vs_mpmath=float(np.abs(fp64_f - mp_f).max()) if m else float("inf"),
        max_vs_zeros_fp64=float(np.abs(fp64_f - zeros_f).max()) if m else float("inf"),
        max_vs_zeros_mpmath=float(genuine),
    )


# ----------------------------------------------------------------------------
# F2: extrapolating the cutoff-sequence
# ----------------------------------------------------------------------------


@dataclass
class AcceleratedZero:
    """Outcome of accelerating one zero's cutoff-sequence toward ``x -> infinity``."""

    index: int  # 1-based zero index
    cutoffs: list  # the prime cutoffs x_j swept
    estimates: list  # nu_index(x_j), the forward sequence
    zero: mp.mpf  # zeta_index (the yardstick, used only here)
    raw_error: mp.mpf  # |nu_index(x_max) - zeta_index| (best single cutoff)
    wynn_error: mp.mpf  # |Wynn-eps extrapolant - zeta_index|
    shanks_error: mp.mpf  # |iterated-Aitken extrapolant - zeta_index|
    gain: mp.mpf  # raw_error / min(wynn_error, shanks_error)


def accelerate_zero(
    index: int,
    eigen_sequence: list,
    cutoffs: list,
    zero,
) -> AcceleratedZero:
    """Apply the accelerators to a forward cutoff-sequence of one eigenvalue.

    ``eigen_sequence[j]`` is ``nu_index`` at cutoff ``cutoffs[j]`` (built upstream
    from the prime-driven operator — forward; this routine never sees a zero until
    the final comparison). Returns the raw best-cutoff error and the Wynn-epsilon /
    iterated-Aitken extrapolant errors against ``zero``. The accelerators consume
    *only* ``eigen_sequence`` — the structural forward guarantee.
    """
    zero = mp.mpf(zero)
    raw = abs(eigen_sequence[-1] - zero)
    wynn = acceleration.wynn_epsilon(eigen_sequence)
    shanks_seq = acceleration.shanks(eigen_sequence, passes=1)
    wynn_err = abs(wynn - zero) if wynn is not None else mp.inf
    shanks_err = abs(shanks_seq[-1] - zero) if shanks_seq else mp.inf
    best = min(wynn_err, shanks_err)
    return AcceleratedZero(
        index=index,
        cutoffs=list(cutoffs),
        estimates=list(eigen_sequence),
        zero=zero,
        raw_error=raw,
        wynn_error=wynn_err,
        shanks_error=shanks_err,
        gain=(raw / best) if best > 0 else mp.inf,
    )
