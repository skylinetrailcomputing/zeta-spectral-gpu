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

import math
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

    def tracking_length(self, *, rel_tol: float = 1e-3) -> int:
        """The forward zero-tracking range ``k*`` of this cell (see
        :func:`tracking_length`)."""
        return tracking_length(self.errors, self.zeros, rel_tol=rel_tol)


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
# The zero-tracking range k*(x) — the universality bridge (#53)
# ----------------------------------------------------------------------------


def tracking_length(errors, zeros, *, rel_tol: float = 1e-3) -> int:
    """Forward zero-tracking range ``k*``: the leading block the operator tracks.

    ``errors[k] = |nu_k - zeta_k|`` and ``zeros[k] = zeta_k`` (0-based), as produced
    by :func:`convergence_errors` / :class:`ccm.CCMResult`. The CCM operator
    reproduces the low ordinates super-exponentially well, then detaches sharply at
    the resolution edge where the zero density outruns the pole spacing ``2 pi / L``
    (#65). ``k*`` is the largest ``k`` such that *every* index ``j < k`` matches its
    zero to relative tolerance — ``|nu_j - zeta_j| / zeta_j < rel_tol`` — i.e. the
    length of the leading contiguous tracked block (robust to a lone spike past the
    break, unlike the first-crossing index). Monotone non-decreasing in ``rel_tol``.
    Returns ``0`` when even ``nu_1`` is off (an under-resolved ``xi``).

    Where #65's ``edge`` ``k_cross`` marks where the error first reaches the *fixed*
    Heisenberg floor ``1/(4 ln lambda)`` at one cutoff, ``k*`` is read at a chosen
    relative tolerance and is meant to be **swept over the cutoff ``x``**: it is the
    quantity behind the #18 observation that the GUE-tracking range extends with the
    prime cutoff (``knowledge/ccm-universality.md``).

    Forward: the zeros enter only to score an already-computed, prime-built spectrum
    — never as input. See the module docstring / ``project-framing.md``.
    """
    n = min(len(errors), len(zeros))
    k = 0
    for j in range(n):
        z = abs(zeros[j])
        if z == 0 or abs(errors[j]) / z >= rel_tol:
            break
        k += 1
    return k


def tracking_height(zeros, k_star: int):
    """The ordinate ``zeta_{k*}`` at the tracking edge (``None`` if ``k* == 0``).

    The companion to :func:`tracking_length`: where ``k*`` counts tracked levels,
    ``t*(x) = zeta_{k*}`` is the *energy* up to which the operator tracks. The
    forward density-balance prediction is that ``t*(x)`` grows **linearly** in the
    prime cutoff ``x`` (poles ``d_n = 2 pi n / L`` of spacing ``2 pi / ln x``
    outnumber the zeros while ``ln(t / 2 pi e) < ln x``, i.e. ``t < 2 pi e x``), so
    ``t*(x) / x`` should plateau near ``2 pi e``. Tested against the measured sweep.
    """
    if k_star <= 0:
        return None
    return zeros[k_star - 1]


# ----------------------------------------------------------------------------
# The gain law: convergence tracks the log-window, not the prime content
# ----------------------------------------------------------------------------


def _primes_upto(n: int) -> list[int]:
    """Primes ``<= n`` by a small sieve (the cutoffs here are ``O(100)``)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def prime_count(c) -> int:
    """``pi(c)`` — the number of primes ``<= c`` (the cutoff's "prime content")."""
    return len(_primes_upto(int(c)))


def prime_power_count(c) -> int:
    """The number of prime powers ``p^m <= c`` — the von Mangoldt support, i.e. the
    operator's *actual* arithmetic content at cutoff ``c`` (via :func:`ccm.prime_powers`)."""
    return len(ccm.prime_powers(c))


def _safe_corr(a, b) -> float:
    """Pearson ``r`` of two same-length sequences; ``nan`` if either is constant or
    there are fewer than two points (a one-step sweep has no correlation)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class GainStep:
    """One step ``c0 -> c1`` of the first-zero gain sweep."""

    c0: int
    c1: int
    err0: mp.mpf  # first-zero error |nu_1 - zeta_1| at c0
    err1: mp.mpf  # ... and at c1
    gain: float  # log10(err0 / err1): orders of magnitude gained on the first zero
    dln_c: float  # ln c1 - ln c0: the log-window growth (L = ln c)
    dpi: int  # pi(c1) - pi(c0): new primes
    dpp: int  # new prime powers (von Mangoldt support change)


@dataclass
class GainLaw:
    """What governs the per-step first-zero gain: the log-window, not the primes (#94).

    The forward fingerprint of the operator's ``L = ln c`` log-window. The per-step
    accuracy gain on the first zero tracks the window growth (``ln c`` / ``d ln c``),
    not the arithmetic content (``d pi`` / new prime powers) — independently
    reproducing Groskin's observation on the ``connes-cvs`` sweep (issue #1 there:
    the largest single-step gain, ``c=13 -> 14``, adds *no* new prime, and the gain
    correlates with ``log c`` at ``r ~ -0.96``, not with prime content). Correlations
    are taken over the steps whose *both* endpoints resolved.

    Forward: each cell is the prime-built operator's first eigenvalue; the zeros enter
    only to score it, after the fact. See the module docstring / ``ccm-convergence-law.md``.
    """

    cutoffs: list  # the integer cutoffs c swept
    N: int
    errors: list  # first-zero error |nu_1 - zeta_1| per cutoff (mpf)
    resolved: list  # per-cutoff resolution flag (first-zero error < resolve_tol)
    steps: list  # list[GainStep] over consecutive resolved pairs only
    r_gain_vs_ln_c: float  # corr(gain, ln c1): vs the log-window LEVEL (Groskin's r)
    r_gain_vs_dln_c: float  # corr(gain, d ln c): vs the window-growth INCREMENT
    r_gain_vs_dpi: float  # corr(gain, d pi): vs new-prime content
    r_gain_vs_dpp: float  # corr(gain, d#prime-powers): vs von Mangoldt support


def _build_gain_steps(cutoffs, errors, resolved) -> list:
    """Per-step gain records over consecutive *resolved* cutoff pairs.

    Shared by :func:`first_zero_gain_law` (first zero) and :func:`per_index_gain_law`
    (each zero ``k``): given one eigenvalue's error cutoff-sequence (``mpf``) and a
    per-cutoff resolution flag, build a :class:`GainStep` for every adjacent pair whose
    *both* endpoints resolved — an unresolved error is precision noise, never a
    finite-cutoff gain. ``dpi``/``dpp`` are cutoff-level, identical across indices.
    """
    steps: list = []
    for i in range(1, len(cutoffs)):
        if not (resolved[i - 1] and resolved[i]):
            continue
        c0, c1 = cutoffs[i - 1], cutoffs[i]
        e0, e1 = errors[i - 1], errors[i]
        with mp.workdps(50):
            gain = float(mp.log10(e0 / e1))
            dln_c = float(mp.log(c1) - mp.log(c0))
        steps.append(
            GainStep(
                c0=c0,
                c1=c1,
                err0=e0,
                err1=e1,
                gain=gain,
                dln_c=dln_c,
                dpi=prime_count(c1) - prime_count(c0),
                dpp=prime_power_count(c1) - prime_power_count(c0),
            )
        )
    return steps


def first_zero_gain_law(
    cutoffs,
    N: int,
    *,
    dps: int | None = None,
    resolve_tol: float = 1e-3,
) -> GainLaw:
    """Sweep the first-zero error over cutoffs and find what governs the per-step gain.

    Forward: for each integer cutoff ``c`` the prime-built operator is assembled at
    ``N`` (and ``dps``, default :func:`suggest_dps`), and its first eigenvalue is
    compared to ``zeta_1`` — the only place a zero enters, after the fact. Over each
    consecutive pair of *resolved* cells the order-of-magnitude gain
    ``log10(err0 / err1)`` is correlated against the log-window (``ln c``, ``d ln c``)
    and against the arithmetic content (``d pi``, ``d#prime-powers``).

    A cell whose first-zero error exceeds ``resolve_tol`` — ``xi`` under-resolved at
    this ``dps`` — is flagged in :attr:`GainLaw.resolved` and never enters a step (an
    un-resolved error is precision noise, not a finite-cutoff gain). Note the
    first-zero error saturates at a finite-``N`` floor for large ``c``, so ``N`` must
    comfortably exceed the cutoff range for the gain to read the cutoff, not the
    truncation.
    """
    cutoffs = [int(c) for c in cutoffs]
    errors: list = []
    resolved: list = []
    for c in cutoffs:
        d = dps if dps is not None else suggest_dps(c)
        res = ccm.converge(N, mp.sqrt(c), 1, dps=d)
        with mp.workdps(d):
            err = res.errors[0]
        errors.append(err)
        resolved.append(bool(err < resolve_tol))

    steps = _build_gain_steps(cutoffs, errors, resolved)
    gains = [s.gain for s in steps]
    return GainLaw(
        cutoffs=cutoffs,
        N=N,
        errors=errors,
        resolved=resolved,
        steps=steps,
        r_gain_vs_ln_c=_safe_corr(gains, [math.log(s.c1) for s in steps]),
        r_gain_vs_dln_c=_safe_corr(gains, [s.dln_c for s in steps]),
        r_gain_vs_dpi=_safe_corr(gains, [s.dpi for s in steps]),
        r_gain_vs_dpp=_safe_corr(gains, [s.dpp for s in steps]),
    )


# ----------------------------------------------------------------------------
# Per-index gain: does the log-window law carry up the band to gamma_k? (#99)
# ----------------------------------------------------------------------------


@dataclass
class IndexGain:
    """The cutoff-gain record for one zero index ``k`` (per-index companion to GainLaw).

    Forward: ``gamma_k`` is the prime-built operator's ``k``-th positive eigenvalue at
    each cutoff; the ordinate ``zeta_k`` only scores it, after the fact. ``mean_gain``
    (mean per-step order-of-magnitude gain over the resolved steps) is the robust
    per-index measure — see :func:`per_index_gain_law`.
    """

    k: int  # 1-based zero index
    gains: list  # per-step gain log10(err0/err1) over resolved cutoff pairs
    mean_gain: float  # mean per-step gain (orders/step); nan if no resolved steps
    n_steps: int  # number of resolved steps contributing
    first_resolved_cutoff: (
        int | None
    )  # smallest cutoff where gamma_k resolves, else None


@dataclass
class PerIndexGainLaw:
    """Per-zero cutoff-gain across a sweep: does the log-window law carry up the band?

    The forward generalisation of :func:`first_zero_gain_law` from the first zero to
    each of the first ``count`` zeros — Groskin's ``connes-cvs`` #1 question (#99): does
    the super-exponential log-window convergence carry up to ``gamma_k``, or does each
    ``gamma_k`` meet its own edge as ``k`` climbs toward ``N``? The robust read is the
    per-index ``mean_gain``: through the resolved bulk every ``gamma_k`` converges
    super-exponentially with ``mean_gain`` decaying *gently and monotonically* in ``k``
    — one law, a mild depth gradient, no second regime within the band. The detachment
    is the ``t*`` tracking edge (:func:`tracking_height`, #53) — a boundary, where the
    secular roots cluster picket-like, not a second convergence law.

    Why ``mean_gain`` and not the gain-vs-cutoff *correlation* :class:`GainLaw` reports:
    the correlation is **not** a robust per-index discriminator — it is dominated by
    finite-``N`` floor deceleration (the late steps slow as the error nears the floor)
    and flips sign between ``N=80`` (floor-decelerated, ``r ~ -0.74``) and ``N=100``
    (floor far below, ``r ~ +0.3``) at the *same* window. ``mean_gain`` is stable
    across ``N``. (Groskin's underlying observation — the largest single steps add no
    new prime — is robust at both ``N``.)
    """

    N: int
    count: int
    cutoffs: list
    per_index: list  # list[IndexGain], k = 1..count
    zeros: list  # zeta_k (the yardstick)


def per_index_gain_law(
    cutoffs,
    N: int,
    *,
    count: int = 10,
    dps: int | None = None,
    resolve_tol: float = 1e-3,
) -> PerIndexGainLaw:
    """Per-zero cutoff-gain sweep — does the log-window law carry up to ``gamma_k``? (#99).

    For each integer cutoff the prime-built operator is assembled at ``N`` (and
    ``dps``, default :func:`suggest_dps`) and its first ``count`` positive eigenvalues
    are scored against the ordinates — the only place a zero enters, after the fact.
    For each index ``k`` the per-step order-of-magnitude gain
    ``log10(|gamma_k(c0)-zeta_k| / |gamma_k(c1)-zeta_k|)`` is formed over the adjacent
    cutoff pairs where ``gamma_k`` is resolved at *both* endpoints (relative error
    ``< resolve_tol`` — a relative gate, since the ordinates grow with ``k``), and
    reduced to the per-index ``mean_gain``.

    The finding (#99): through the resolved bulk ``mean_gain(k)`` decays gently and
    monotonically in ``k`` — the super-exponential log-window convergence carries
    uniformly up the band, the gentle gradient being the in-band precursor of the
    ``t*`` detachment, not a second law. ``first_resolved_cutoff`` is the complementary
    edge probe: a zero that only resolves at a larger cutoff is meeting its own edge; in
    a bulk window every index resolves at every cutoff. See
    ``knowledge/ccm-convergence-law.md``.
    """
    cutoffs = [int(c) for c in cutoffs]
    # Per-cutoff: the full error vector (k = 0..count-1) of the prime-built spectrum.
    err_rows: list = []
    zeros: list = []
    for c in cutoffs:
        d = dps if dps is not None else suggest_dps(c)
        res = ccm.converge(N, mp.sqrt(c), count, dps=d)
        with mp.workdps(d):
            err_rows.append([abs(e) for e in res.errors])
        if not zeros:
            zeros = list(res.zeros)

    per_index: list = []
    for k in range(count):
        errs_k = [row[k] if k < len(row) else mp.inf for row in err_rows]
        with mp.workdps(50):
            resolved_k = [bool(e / abs(zeros[k]) < resolve_tol) for e in errs_k]
        steps = _build_gain_steps(cutoffs, errs_k, resolved_k)
        gains = [s.gain for s in steps]
        first_res = next((cutoffs[i] for i, r in enumerate(resolved_k) if r), None)
        per_index.append(
            IndexGain(
                k=k + 1,
                gains=gains,
                mean_gain=float(np.mean(gains)) if gains else float("nan"),
                n_steps=len(steps),
                first_resolved_cutoff=first_res,
            )
        )
    return PerIndexGainLaw(
        N=N, count=count, cutoffs=cutoffs, per_index=per_index, zeros=zeros
    )


# ----------------------------------------------------------------------------
# The precision-artifact quantifier (the headline Phase-0 finding)
# ----------------------------------------------------------------------------


def _fp64_even_eigenvector(N: int, lam) -> tuple[np.ndarray, float]:
    """The fp64 near-null even eigenvector ``xi`` of ``QW_lambda^N`` (the corrupted
    one the fp64 path yields), and its ``|eigenvalue|`` ``eps_fp64``.

    ``eigh`` of the fp64-assembled Weil matrix, take the smallest-``|eigenvalue|``
    vector, symmetrise to the even subspace (Def. 5.3) and normalise. Returns the
    length-``2N+1`` numpy array (indexed ``i -> n = i - N``) and the corresponding
    smallest ``|eigenvalue|``. fp64 only (numpy ``eigh``); no GPU. The fp64 ``eps_N``
    underflows almost immediately, so beyond the smallest cutoffs this ``xi`` is
    roundoff — the whole point of #65/#82.
    """
    from . import ccm_gpu

    A = ccm_gpu.assemble_weil_matrix_fp64(N, float(lam))
    w, V = np.linalg.eigh(A)
    i = int(np.argmin(np.abs(w)))
    v = V[:, i]
    even = np.array([(v[N + n] + v[N - n]) / 2.0 for n in range(-N, N + 1)])
    return even / np.linalg.norm(even), float(w[i])


def _pole_gap(root, L) -> int:
    """Index ``n`` of the pole gap ``(d_n, d_{n+1})`` a positive ``root`` sits in.

    ``d_n = 2 pi n / L``; the gap index is ``floor(root * L / (2 pi))``. Tagging
    every root by its gap is the robust alignment diagnostic for #82: near the
    resolution edge the fp64 and mpmath roots land in the **same** gap (both pinned
    to the bulk pole ``d_n``), the structural sign that the edge is robust to the
    ``xi``-corruption that scrambles the low band.
    """
    return int(mp.floor(mp.mpf(root) * mp.mpf(L) / (2 * mp.pi)))


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
    lam = mp.mpf(lam)
    if dps is None:
        dps = suggest_dps(lam**2)
    L = 2 * mp.log(lam)

    # Genuine spectrum (mpmath) and zeros.
    mp_spec = ccm.operator_spectrum(N, lam, count=count, dps=dps)
    with mp.workdps(dps):
        zeros = ccm.reference_ordinates(count)

    # fp64 xi -> secular roots (root-find kept in mpmath so only xi differs).
    even, eps_fp64 = _fp64_even_eigenvector(N, lam)
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
        eps_fp64=eps_fp64,
        max_vs_mpmath=float(np.abs(fp64_f - mp_f).max()) if m else float("inf"),
        max_vs_zeros_fp64=float(np.abs(fp64_f - zeros_f).max()) if m else float("inf"),
        max_vs_zeros_mpmath=float(genuine),
    )


# ----------------------------------------------------------------------------
# Edge robustness: where does fp64 xi-corruption actually land? (#82)
# ----------------------------------------------------------------------------


@dataclass
class EdgeCorruption:
    """Per-index fp64-vs-mpmath spectrum comparison: low band vs resolution edge.

    The #82 spike. :func:`fp64_spectrum_corruption` reports only the *max* over a
    low set; this resolves the question that gates any Sliwinski outreach — is the
    fp64 corruption confined to the **low / near-null** eigenvalues, or does it also
    hit the **edge** eigenvalues that dominate Sliwinski's uniform error
    ``E = max_k |nu_k - zeta_k|`` (Conjecture 4.1)?

    Three per-index arrays (length :attr:`count`, 0-based ``k``), as floats (the
    genuine values reach ``~1e-55`` — far below float's relative precision, but the
    *magnitude* survives, which is all a low-vs-edge comparison needs; the deep
    digits only matter for the super-exponential acceleration study, not here):

    - :attr:`genuine` ``= |nu_k^{mpmath} - zeta_k|`` — the true finite-cutoff error;
    - :attr:`fp64_error` ``= |nu_k^{fp64} - zeta_k|`` — what a fp64 run reports;
    - :attr:`corruption` ``= |nu_k^{fp64} - nu_k^{mpmath}|`` — pure ``xi``-corruption.

    Plus the pole-gap each root sits in (:attr:`gap_mpmath` / :attr:`gap_fp64`):
    where they agree the fp64 and mpmath roots are pinned to the *same* bulk pole
    ``d_n = 2 pi n / L`` (the edge-robustness signature). :attr:`k_floor` is the
    resolution-edge index — the first ``k`` whose genuine error reaches the
    Heisenberg floor :attr:`bound` ``= 1/(4 ln lambda)``.
    """

    N: int
    cutoff: mp.mpf
    dps: int
    count: int
    genuine: list  # |nu_k^{mpmath} - zeta_k|
    fp64_error: list  # |nu_k^{fp64} - zeta_k|
    corruption: list  # |nu_k^{fp64} - nu_k^{mpmath}|
    gap_mpmath: list  # pole gap of each mpmath root
    gap_fp64: list  # pole gap of each fp64 root
    bound: float  # 1/(4 ln lambda)
    k_floor: int | None  # first k whose genuine error reaches the floor


def edge_corruption_profile(
    N: int, lam, *, count: int | None = None, dps: int | None = None
) -> EdgeCorruption:
    """Per-index genuine error / fp64 error / ``xi``-corruption across the spectrum.

    Computes the genuine (mpmath) spectrum and the fp64-``xi`` spectrum through the
    *same* secular root-finder (so only ``xi`` differs), compares both to the zeros
    index-by-index, and tags every root by its pole gap. ``count`` defaults to ``N``
    (the full first-``N`` set Thm 3.1 / Conj 4.1 are stated over). Forward: the
    spectra are the prime-built operator's; the zeros only score them, after the
    fact (see the module docstring).

    The #82 finding (read off the bands via :func:`summarize_edge_bands`): the
    corruption is **largest in the low / near-null band and decays toward the edge**,
    where the eigenvalues are pinned to the bulk poles ``d_n`` and are robust to the
    ``xi``-corruption. So a fp64 *uniform-error* measurement is edge-dominated and
    plausibly genuine once the genuine edge error (growing ``~ zeta_N``) overtakes
    the bounded low-band corruption.
    """
    lam = mp.mpf(lam)
    if count is None:
        count = N
    if dps is None:
        dps = suggest_dps(lam**2)
    L = 2 * mp.log(lam)

    mp_spec = ccm.operator_spectrum(N, lam, count=count, dps=dps)
    even, _eps_fp64 = _fp64_even_eigenvector(N, lam)
    with mp.workdps(50):
        fp64_roots = ccm.operator_eigenvalues(
            [mp.mpf(float(c)) for c in even], N, L, count
        )
    with mp.workdps(dps):
        zeros = ccm.reference_ordinates(count)
        m = min(len(mp_spec), len(fp64_roots), len(zeros))
        # Genuine error in mpmath (it is ~1e-55, below float's relative precision at
        # the O(100) eigenvalue magnitude); cast to float only for storage.
        genuine = [float(abs(mp_spec[k] - zeros[k])) for k in range(m)]
        fp64_error = [float(abs(fp64_roots[k] - zeros[k])) for k in range(m)]
        corruption = [float(abs(fp64_roots[k] - mp_spec[k])) for k in range(m)]
        gap_mpmath = [_pole_gap(mp_spec[k], L) for k in range(m)]
        gap_fp64 = [_pole_gap(fp64_roots[k], L) for k in range(m)]
        bound = float(heisenberg_bound(lam))
    k_floor = next((k for k in range(m) if genuine[k] >= bound), None)
    return EdgeCorruption(
        N=N,
        cutoff=lam**2,
        dps=dps,
        count=m,
        genuine=genuine,
        fp64_error=fp64_error,
        corruption=corruption,
        gap_mpmath=gap_mpmath,
        gap_fp64=gap_fp64,
        bound=bound,
        k_floor=k_floor,
    )


def summarize_edge_bands(prof: EdgeCorruption, *, k_split: int | None = None) -> dict:
    """Split an :class:`EdgeCorruption` into low (``k < k_split``) vs edge bands.

    ``k_split`` defaults to ``prof.k_floor`` (the resolution edge — where the genuine
    error reaches the Heisenberg floor). Returns, per band, the max corruption /
    genuine / fp64 error, plus the global maxima with their argmax index. The #82
    crossover readout: ``e_argmax_band`` is ``"low"`` while the bounded low-band
    corruption sets fp64's uniform error ``E`` (small ``N``), and ``"edge"`` once the
    genuine resolution edge overtakes it (large ``N``, Sliwinski's regime) — at which
    point the fp64 ``E`` measurement is edge-dominated and plausibly genuine.
    """
    n = prof.count
    if k_split is None:
        k_split = prof.k_floor if prof.k_floor is not None else n // 2
    k_split = max(1, min(k_split, n))
    cor = np.array(prof.corruption)
    gen = np.array(prof.genuine)
    fpe = np.array(prof.fp64_error)

    def band(sl):
        return {
            "max_corruption": float(cor[sl].max()) if cor[sl].size else 0.0,
            "max_genuine": float(gen[sl].max()) if gen[sl].size else 0.0,
            "max_fp64_error": float(fpe[sl].max()) if fpe[sl].size else 0.0,
        }

    e_arg = int(fpe.argmax())
    return {
        "k_split": k_split,
        "count": n,
        "low": band(slice(0, k_split)),
        "edge": band(slice(k_split, n)),
        "E_fp64": float(fpe.max()),  # what a fp64 run would report
        "E_genuine": float(gen.max()),  # the true uniform error
        "E_argmax": e_arg,
        "E_argmax_band": "low" if e_arg < k_split else "edge",
        "max_corruption": float(cor.max()),
        "max_corruption_index": int(cor.argmax()),
    }


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
