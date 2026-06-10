"""Intermediate statistics (Seba / rank-one) of the CCM pole-locked tail (#87).

The CCM operator ``D_log^{(lambda,N)}`` is a rank-one perturbation of a scaling
operator, so its spectrum solves the secular equation (``ccm.operator_eigenvalues``)

    F(z) = sum_{n=-N}^{N} xi_n / (d_n - z) = 0,    d_n = 2 pi n / L,

a *picket* of poles with spacing ``Delta = 2 pi / L`` and signed, prime-built
couplings ``xi_n`` (the minimal even eigenvector of the Weil form). Quantum chaos
has a named theory for exactly this operator class — point scatterers (Seba 1990)
and the intermediate spectral statistics of their rank-one secular equation
(Bogomolny–Gerland–Schmit 1999) — with two deviations here worth keeping explicit:

- the unperturbed levels are a **picket** (uniform poles), not Poisson, so the
  strong-coupling endpoint is *not* the BGS semi-Poisson law (that needs Poisson
  poles); semi-Poisson enters only as a reference marker between Poisson and GUE;
- the couplings are **signed** (eigenvector components, not ``|c|^2 > 0``), so the
  roots need not interlace the poles — a gap can hold 0, 1 or 2 roots, exactly
  what the root-finder in :mod:`ccm` observes.

The theory is **locality**: the position of a root is set by the couplings of the
few poles around it, measured against the pole spacing. Two levels of it:

- the **weak-coupling (pinning) expansion** — a root attaches to pole ``d_n`` at

      delta_n = xi_n / R_n,    R_n = sum_{m != n} xi_m / (d_m - d_n),

  (``R_n`` = the smooth background of ``F`` at the pole), valid while the
  dimensionless coupling ``w_n = |delta_n| / Delta`` is small
  (:func:`pinned_tail`);
- the **local two-pole model** — keep the gap's two flanking pole terms exactly
  and freeze the rest of ``F`` at its mid-gap value, making the secular equation
  a quadratic per gap (:func:`local_gap_model`). It predicts both the root
  *positions* and the **gap occupancy** (0, 1 or 2 roots — the signed couplings
  break interlacing), with no root-finding on the full ``F``.

Both are parameter-free functions of the couplings. The measured CCM profile
turns out to sit in the *intermediate* regime — ``w_n`` fluctuates at O(0.1..1)
down the whole tail because the background ``R_n`` decays at the same rate as the
couplings — so the pole-locked tail is a picket with O(1)-correlated jitter
rather than the trivially-pinned ``w -> 0`` limit, and the local model (not the
first-order formula) is the right statistics predictor. Of the two boundaries in
``knowledge/ccm-universality.md``, the occupancy deficit (:func:`deficit_plateau`)
reads off the **density crossover** ``t ~ 2 pi x``; the **tracking height**
``t* ~ 12 x`` (#53) turns out to be invisible to these local readouts — see the
knowledge note for both results.

Forward throughout: everything here consumes the operator's own couplings and
poles; no zeta zero is input anywhere (the zeros only score the crossover against
the #53 law, after the fact). See ``knowledge/ccm-intermediate-statistics.md``.

Precision: the background sums ``R_n`` / ``B`` cancel down from O(0.1)-size
low-mode terms to the scale of the *local* couplings (1e-25 and below in the deep
tail) — float64 returns pure roundoff there, the same wall as everywhere else in
the CCM stack (#65). So ``xi`` must come from the mpmath eigensolve **and** the
sums here run in mpmath at the caller's working precision (wrap in
``mp.workdps`` matching the ``xi`` cache); only the O(1)-conditioned outputs
(offsets over ``Delta``, root positions) are cast to float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

# Reference values for the folded spacing ratio <r~> (see spacing.py for the
# Poisson/GOE/GUE/GSE values). Semi-Poisson = every other level of a Poisson
# process (BGS 1999): consecutive spacings are then *independent* Gamma(2)
# variables, the ratio r = s/s' has density 6r/(1+r)^4, and
# <r~> = 2 int_0^1 6 r^2/(1+r)^4 dr = 1/2 exactly (asserted against a simulated
# decimated Poisson process in the tests). A perfectly rigid picket gives 1.
MEAN_RATIO_SEMI_POISSON = 0.5
MEAN_RATIO_PICKET = 1.0


def fold_even_couplings(xi, N: int) -> np.ndarray:
    """Couplings ``xi_n`` at the positive poles ``d_1..d_N`` from the full vector.

    ``xi`` is the length-``2N+1`` even eigenvector (index ``i -> n = i - N``, the
    :mod:`ccm` convention; mpf or float entries). Returns float64 ``[xi_1..xi_N]``
    — the effective coupling at pole ``d_n`` is the *unfolded* component ``xi_n``
    (the partial-fraction residue of ``F`` at ``d_n``), not the parity-doubled one.
    """
    if len(xi) != 2 * N + 1:
        raise ValueError(f"xi must have length 2N+1 = {2 * N + 1}, got {len(xi)}")
    return np.array([float(xi[N + n]) for n in range(1, N + 1)], dtype=np.float64)


@dataclass
class PinnedTail:
    """The weak-coupling (pinning) read of one cell's secular equation.

    Arrays are indexed by positive pole number ``n = 1..N`` (0-based slot
    ``n - 1``). ``offsets`` can be invalid (``> Delta/2`` in magnitude) where the
    expansion has broken down — :attr:`pinned` masks the usable region.
    """

    N: int
    L: float
    poles: np.ndarray  # d_n = 2 pi n / L, n = 1..N
    couplings: np.ndarray  # xi_n at the positive poles
    background: np.ndarray  # R_n = sum_{m != n} xi_m / (d_m - d_n)
    offsets: np.ndarray  # delta_n = xi_n / R_n (first-order pinning)
    w: np.ndarray  # |delta_n| / Delta, the dimensionless coupling
    pinned: np.ndarray  # w < 1/2: the expansion's validity mask

    @property
    def spacing(self) -> float:
        """The pole spacing ``Delta = 2 pi / L``."""
        return 2.0 * np.pi / self.L


def pinned_tail(xi, N: int, L) -> PinnedTail:
    """First-order pinning analysis of the secular equation for one ``(xi, N, L)``.

    ``xi`` is the full even eigenvector (length ``2N+1``, mpf entries from the
    mpmath eigensolve — see :func:`fold_even_couplings`). The background sum runs
    over *all* modes ``m in [-N, N], m != n`` — including ``m = 0`` and the mirror
    ``m = -n`` — folded over parity, and is evaluated in mpmath at the caller's
    working precision (the sum cancels far below float64; see the module
    docstring). Only the conditioned ratios are returned as float64.
    """
    Lm = mp.mpf(L)
    xi0 = mp.mpf(xi[N])
    xs_mp = [mp.mpf(xi[N + n]) for n in range(1, N + 1)]
    d_mp = [2 * mp.pi * n / Lm for n in range(1, N + 1)]
    d2 = [dn**2 for dn in d_mp]

    # R_n = -xi_0/d_n + sum_{m>=1, m!=n} 2 d_n xi_m/(d_m^2 - d_n^2) - xi_n/(2 d_n)
    # (the unfolded sum_{m != n} xi_m/(d_m - d_n) folded over parity; the last
    # term is the mirror m = -n).
    delta = np.empty(N, dtype=np.float64)
    background = np.empty(N, dtype=np.float64)
    for j in range(N):
        dn = d_mp[j]
        R = mp.fsum(2 * dn * xs_mp[i] / (d2[i] - d2[j]) for i in range(N) if i != j)
        R += -xi0 / dn - xs_mp[j] / (2 * dn)
        background[j] = float(R)
        delta[j] = float(xs_mp[j] / R) if R != 0 else np.inf

    L = float(Lm)
    w = np.abs(delta) * L / (2.0 * np.pi)
    return PinnedTail(
        N=N,
        L=L,
        poles=2.0 * np.pi * np.arange(1, N + 1) / L,
        couplings=fold_even_couplings(xi, N),
        background=background,
        offsets=delta,
        w=w,
        pinned=w < 0.5,
    )


@dataclass
class LocalGapModel:
    """The two-pole local prediction, gap by gap.

    Gap ``g`` (0-based, ``g = 0..N-1``) is ``(d_g, d_{g+1})`` with ``d_0 = 0``
    (the ``xi_0 / (0 - z)`` pole). :attr:`occupancy` counts the quadratic's real
    roots inside each open gap; :attr:`levels` is the sorted union — the local
    theory's whole positive spectrum, built from couplings with no root-finding
    on the full secular function.
    """

    N: int
    L: float
    edges: np.ndarray  # d_0 = 0, d_1, ..., d_N (length N + 1)
    occupancy: np.ndarray  # roots per gap (length N)
    levels: np.ndarray  # all predicted roots, sorted

    @property
    def spacing(self) -> float:
        return 2.0 * np.pi / self.L


def local_gap_model(xi, N: int, L) -> LocalGapModel:
    """Per-gap quadratic roots of the secular equation, frozen-background.

    In gap ``(d_g, d_{g+1})`` keep the two flanking pole terms of ``F`` exactly
    and evaluate everything else at the gap midpoint (constant ``B``):

        xi_g / (d_g - z) + xi_{g+1} / (d_{g+1} - z) + B = 0,

    a quadratic in ``z`` — the Seba / BGS secular-equation locality argument,
    adapted to signed couplings (which is what allows 0- and 2-root gaps). The
    couplings here are the partial-fraction residues at the positive poles plus
    ``xi_0`` at ``z = 0``; the mirror poles ``-d_n`` are part of the background.
    """
    Lm = mp.mpf(L)
    # Residues: xi_0 at d_0 = 0, xi_n at d_n (and at the mirror -d_n).
    res = [mp.mpf(xi[N + n]) for n in range(0, N + 1)]
    edges_mp = [2 * mp.pi * g / Lm for g in range(0, N + 1)]
    all_poles = [(edges_mp[m], res[m]) for m in range(0, N + 1)] + [
        (-edges_mp[m], res[m]) for m in range(1, N + 1)
    ]

    occupancy = np.zeros(N, dtype=np.int64)
    found: list[float] = []
    for g in range(N):
        lo, hi = edges_mp[g], edges_mp[g + 1]
        mid = (lo + hi) / 2
        # The frozen background: F(mid) minus the two flanking pole terms. The
        # sum cancels to the local coupling scale — mpmath territory (#65).
        B = mp.fsum(r / (dm - mid) for dm, r in all_poles)
        B -= res[g] / (lo - mid) + res[g + 1] / (hi - mid)
        c_lo, c_hi = res[g], res[g + 1]
        # xi_lo (hi - z) + xi_hi (lo - z) + B (lo - z)(hi - z) = 0
        a = B
        b = -B * (lo + hi) - (c_lo + c_hi)
        c = B * lo * hi + c_lo * hi + c_hi * lo
        if a != 0:
            disc = b * b - 4 * a * c
            if disc < 0:
                roots = []
            else:
                s = mp.sqrt(disc)
                roots = [(-b + s) / (2 * a), (-b - s) / (2 * a)]
        elif b != 0:
            roots = [-c / b]
        else:
            roots = []
        for z in roots:
            if lo < z < hi:
                occupancy[g] += 1
                found.append(float(z))
    return LocalGapModel(
        N=N,
        L=float(Lm),
        edges=np.array([float(e) for e in edges_mp]),
        occupancy=occupancy,
        levels=np.sort(found),
    )


@dataclass
class DeficitPlateau:
    """The cumulative root-vs-pole deficit's plateau — the density crossover."""

    deficit_max: int  # peak of (#gaps - #predicted roots) up to gap g
    t_first: float  # ordinate where the peak is first reached
    t_mid: float  # plateau midpoint — the headline crossover estimate
    t_last: float  # ordinate where the deficit last sits at its peak


def deficit_plateau(model: LocalGapModel) -> DeficitPlateau | None:
    """The density crossover, read from the predicted occupancy alone.

    Below ``t ~ 2 pi x`` the operator's root density (which tracks the zero
    density where it can) is *smaller* than the uniform pole density ``L / 2 pi``,
    so empty gaps accumulate and the cumulative deficit ``g - sum(occupancy[:g])``
    grows; above it the densities cross and 2-root gaps stop the growth. The
    deficit therefore peaks on a plateau straddling the density crossover — a
    coupling-side, zero-free read of the ``2 pi x`` line (*not* of the tracking
    height ``t* ~ 12 x``, which lives above it; see the knowledge note). Returns
    ``None`` when no deficit ever accumulates.
    """
    cum = np.cumsum(model.occupancy)
    deficit = np.arange(1, model.N + 1) - cum
    dmax = int(deficit.max())
    if dmax <= 0:
        return None
    idx = np.nonzero(deficit == dmax)[0]
    t_first = float(model.edges[idx[0] + 1])
    t_last = float(model.edges[idx[-1] + 1])
    return DeficitPlateau(
        deficit_max=dmax,
        t_first=t_first,
        t_mid=0.5 * (t_first + t_last),
        t_last=t_last,
    )


def windowed_rtilde(levels: np.ndarray, *, window: int, step: int) -> np.ndarray:
    """Sliding-window mean spacing ratio over a sorted spectrum.

    Returns rows ``(center, mean_rtilde)``: the window-center *level value* (so
    curves from spectra with different level counts share an ordinate axis) and
    the mean ``r~`` over the ``window`` consecutive levels starting every ``step``
    indices. Unfolding-free, like :func:`spacing.spacing_ratios` it wraps.
    """
    from . import spacing

    x = np.sort(np.asarray(levels, dtype=np.float64))
    if window < 4 or x.size < window:
        return np.empty((0, 2))
    rows = []
    for lo in range(0, x.size - window + 1, step):
        block = x[lo : lo + window]
        rows.append(
            (float(block.mean()), float(np.nanmean(spacing.spacing_ratios(block))))
        )
    return np.array(rows)


def semi_poisson_levels(count: int, rng: np.random.Generator) -> np.ndarray:
    """A semi-Poisson spectrum: every other level of a Poisson process (BGS 1999).

    The reference construction behind :data:`MEAN_RATIO_SEMI_POISSON` — consecutive
    spacings are sums of two fresh exponentials, hence independent Gamma(2).
    Used by the tests to verify the surmise-level constant, and available to
    scripts as a plotted reference ensemble.
    """
    gaps = rng.exponential(size=2 * count).reshape(count, 2).sum(axis=1)
    return np.cumsum(gaps)
