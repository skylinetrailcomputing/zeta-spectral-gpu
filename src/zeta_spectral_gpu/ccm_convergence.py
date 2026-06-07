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
