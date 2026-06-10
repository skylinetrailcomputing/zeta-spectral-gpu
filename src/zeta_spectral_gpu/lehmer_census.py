"""Small-gap / Lehmer-pair census at height via the Riemann-Siegel evaluator (#86).

The first *science consumer* of the #55 ``Z(t)`` tool: harvest long runs of
critical-line zero ordinates from sign changes of the Hardy function (primes /
analysis in, zeros out -- forward), then read two things off the consecutive-gap
list:

- the **small-gap tail** of the normalized nearest-neighbour spacing
  distribution, against the GUE level-repulsion law ``p(s) -> (pi^2/3) s^2`` as
  ``s -> 0`` (the sine-kernel coefficient; the Wigner surmise's ``32/pi^2`` is
  the 2x2 approximation of the same repulsion);
- a **Lehmer-pair census** under the Csordas-Smith-Varga criterion, each
  qualifying pair yielding a forward-computed lower bound on the De Bruijn-
  Newman constant ``Lambda`` (pinned to 0 under RH by Rodgers-Tao; see
  ``knowledge/debruijn-newman-flow.md``).

CSV criterion (Csordas-Smith-Varga 1994, as restated by Stopple 2017 in zeta-
ordinate coordinates): for consecutive simple positive zeros ``g- < g+`` of
``Xi(t)`` put ``Delta = g+ - g-`` and

    g = sum_{gamma != g-, g+}  1/(gamma - g-)^2 + 1/(gamma - g+)^2,

the sum over *all other* zeros of ``Xi`` (so both ``+gamma_j`` and the mirror
``-gamma_j``). The pair is a **Lehmer pair** iff ``Delta^2 g < 4/5``, and then

    lambda(Delta, g) = ((1 - 5 Delta^2 g / 4)^{4/5} - 1) / (8 g)  <=  Lambda

in the flow normalization where the zeros of the time-zero function sit at the
ordinates ``gamma`` themselves. The classical DBN literature (the published
``-4.379e-6`` bound of CSV, de Bruijn's ``Lambda <= 1/2``, Polymath15's
``Lambda <= 0.22`` -- and this repo's #20 ``H_t``) parametrizes the *same* flow
through ``H_t(z) ~ Xi_{t/4}(z/2)`` (zeros at ``z = 2 gamma``), so bounds scale
by exactly 4: ``lambda_classical = 4 * lambda``. ``Delta^2 g`` is dimensionless
and identical in both. :func:`csv_lambda` returns the classical normalization
(the one every published table uses); the factor is pinned against the COSV
1993 pair's published bound (see :func:`csv_lambda`), and the end-to-end
pipeline is validated on the classical Lehmer pair near ``t ~ 7005``.

The censused list is finite, so ``g`` is summed over the censused window and
completed with an analytic tail: outside the window the zero ordinates are
replaced by their smooth Riemann-von Mangoldt density ``rho(x) =
log(x/2pi)/2pi``, whose integral against ``1/(x - c)^2`` has a closed form
(:func:`_density_integral`). The tail (and the entire mirror ``-gamma`` axis)
contributes ``O(rho / dist)`` -- per mille of ``g`` for the window sizes used
here -- and it is a *smooth-density estimate*, so the per-pair ``lambda`` is a
measured census readout, not a certified bound (certification would need the
rigorous tail bracketing of the CSV-era papers; the shortlist is small enough
that mpmath polish, :func:`polish_pair`, pins the pair itself to full
precision).

Precision budget (house rule -- written down *before* scanning): the fp64
Riemann-Siegel ``Z`` carries a phase-rounding error growing like ``t``
(:func:`phase_error_model`; the #55 ceiling, ~1e-7 by ``t = 1e8``). A close
pair with normalized gap ``s`` is detectable while the interior extremum of
``Z`` between the two zeros -- ``~ (pi^2/8) A s^2`` for local amplitude scale
``A`` -- clears that noise, giving a resolvable-gap floor ``s_min ~
sqrt(8 k eps / (pi^2 A))`` (:func:`gap_resolution_floor`): ~2e-4 at ``t = 1e6``
and still only ~2e-3 at ``1e8``, far below the smallest gaps a feasible census
will contain (the GUE cube law makes ``P(s < 0.01)`` one in a million zeros) --
so heights up to ``1e7``-plus are comfortably inside budget and the binding
constraint is scan volume, not precision.

Forward, not inverse: the zeros consumed by every statistic here are *produced*
by the Riemann-Siegel scan itself. Known Lehmer pairs from the literature and
``mpmath`` zeros appear only as after-the-fact checks in the tests and script,
never as input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import riemann_siegel
from .zeros import smooth_count

TWO_PI = 2.0 * np.pi
EPS64 = float(np.finfo(np.float64).eps)

# Lehmer-pair threshold on Delta^2 g (CSV 1994, Theorem 1 / Stopple eq. 3).
CSV_THRESHOLD = 0.8


def mean_gap(t: np.ndarray | float) -> np.ndarray | float:
    """Mean zero gap ``2 pi / log(t / 2pi)`` at height ``t`` (Riemann-von Mangoldt)."""
    t = np.asarray(t, dtype=np.float64)
    out = TWO_PI / np.log(t / TWO_PI)
    return out if out.ndim else float(out)


def zero_density(t: np.ndarray | float) -> np.ndarray | float:
    """Smooth zero density ``rho(t) = log(t / 2pi) / 2pi`` -- d/dt of `smooth_count`."""
    t = np.asarray(t, dtype=np.float64)
    out = np.log(t / TWO_PI) / TWO_PI
    return out if out.ndim else float(out)


def phase_error_model(t: np.ndarray | float) -> np.ndarray | float:
    """Model of the fp64 ``Z(t)`` evaluation error at height ``t``.

    The main-sum phases ``theta(t) - t log n`` are ``O(t log t)`` before
    reduction, so fp64 rounding leaves an absolute phase error ``~ eps * t *
    log``; the cosine sum transfers it to ``Z`` at the local amplitude scale.
    The constant is calibrated to the #55 measurement (~1e-7 absolute error at
    ``t = 1e8``); the run script re-measures against ``mpmath.siegelz`` so the
    model is checked, not trusted.
    """
    t = np.asarray(t, dtype=np.float64)
    out = 0.5 * EPS64 * t * np.log(t / TWO_PI)
    return out if out.ndim else float(out)


def gap_resolution_floor(
    t: np.ndarray | float,
    *,
    z_scale: float = 1.0,
    z_err: np.ndarray | float | None = None,
    safety: float = 8.0,
) -> np.ndarray | float:
    """Smallest resolvable normalized gap ``s_min`` at height ``t`` (the budget).

    Between the two zeros of a close pair with normalized gap ``s``, ``Z``
    reaches an interior extremum ``|Z_ext| ~ |Z''| Delta^2 / 8 ~ (pi^2/8) A
    s^2`` (one oscillation spans one mean gap, so ``|Z''| ~ (pi / mean_gap)^2 A``
    with ``A`` the local amplitude scale). The pair's two sign changes survive
    fp64 while ``|Z_ext| > safety * z_err``, i.e. down to

        s_min = sqrt(8 * safety * z_err / (pi^2 * A)).

    ``z_scale`` is ``A`` (pass the window's measured RMS ``Z``; 1.0 is
    conservative -- the RMS grows like ``sqrt(log)``), ``z_err`` defaults to
    :func:`phase_error_model`.
    """
    err = phase_error_model(t) if z_err is None else np.asarray(z_err, np.float64)
    out = np.sqrt(8.0 * safety * err / (np.pi**2 * z_scale))
    return out if np.ndim(out) else float(out)


@dataclass
class CensusWindow:
    """One scanned height window: the harvested zeros plus scan health readouts."""

    t_lo: float
    t_hi: float
    zeros: np.ndarray  # ascending ordinates located in (t_lo, t_hi)
    expected_count: float  # smooth (theta-based) zero count for the window
    rms_z: float  # RMS of Z on the scan grid (the amplitude scale A)
    step: float  # coarse grid step used
    rescued: np.ndarray  # zeros found only by the dip rescan (subset of `zeros`)
    near_misses: np.ndarray  # (t, |Z|) dips that refined to no sign change


def _bisect_sign_changes(
    z_of_t, lo: np.ndarray, hi: np.ndarray, *, iters: int = 40
) -> np.ndarray:
    """Vectorised bisection of bracketed sign changes of a batch evaluator.

    Same scheme as the Davenport-Heilbronn locator; every iteration is one
    batched ``z_of_t`` call, so a GPU evaluator refines all brackets in
    parallel.
    """
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    if lo.size == 0:
        return lo
    z_lo = np.asarray(z_of_t(lo))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        z_mid = np.asarray(z_of_t(mid))
        left = np.signbit(z_lo) != np.signbit(z_mid)
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        z_lo = np.where(left, z_lo, z_mid)
    return 0.5 * (lo + hi)


def scan_zeros(
    t_lo: float,
    t_hi: float,
    *,
    evaluator=None,
    step_fraction: float = 0.05,
    fine_fraction: float = 0.002,
    dip_threshold: float = 0.25,
    refine_iters: int = 40,
) -> CensusWindow:
    """Harvest the on-line zero ordinates in ``(t_lo, t_hi)`` from a ``Z`` scan.

    Three resolution tiers, each handing the next a tiny candidate set:

    1. coarse grid (``step_fraction`` mean gaps): sign changes bracket every
       pair with normalized gap ``s >~ step_fraction``; vectorised bisection
       refines them;
    2. **dip rescan**: interior local minima of ``|Z|`` with same-sign
       neighbours and ``|Z| < dip_threshold * RMS`` are exactly where a close
       pair (or a Lehmer-phenomenon near-miss) hides between coarse nodes; each
       candidate cell is re-gridded at ``fine_fraction`` mean gaps, recovering
       pairs down to ``s ~ fine_fraction``;
    3. dips that still show no sign change are returned as ``near_misses`` for
       optional mpmath polish (:func:`polish_pair` handles the found pairs).

    ``evaluator`` is any batch callable ``t -> Z(t)`` (default
    :func:`riemann_siegel.hardy_z`; pass ``riemann_siegel_gpu.hardy_z_gpu`` for
    the GPU scan -- the house CPU-reference rule is a test asserting both give
    the same census). The returned ``expected_count`` is the smooth theta-based
    count; the actual count fluctuates around it by ``S(t)`` (O(1) at these
    heights), so a larger deficit flags missed zeros.
    """
    ev = evaluator if evaluator is not None else riemann_siegel.hardy_z
    delta = float(mean_gap(t_hi))  # smallest mean gap in the window: conservative
    step = step_fraction * delta
    grid = np.arange(t_lo, t_hi + step, step)
    z = np.asarray(ev(grid))
    rms = float(np.sqrt(np.mean(z * z)))

    flip = np.signbit(z[:-1]) != np.signbit(z[1:])
    zeros = _bisect_sign_changes(
        ev, grid[:-1][flip], grid[1:][flip], iters=refine_iters
    )

    # Dip rescan: |Z| local minima strictly inside a same-sign triple.
    sgn = np.signbit(z)
    az = np.abs(z)
    i = np.arange(1, z.size - 1)
    candidate = (
        (sgn[i - 1] == sgn[i])
        & (sgn[i] == sgn[i + 1])
        & (az[i] <= az[i - 1])
        & (az[i] <= az[i + 1])
        & (az[i] < dip_threshold * rms)
    )
    idx = i[candidate]
    rescued = np.empty(0, dtype=np.float64)
    near_misses = np.empty((0, 2), dtype=np.float64)
    if idx.size:
        fine_step = fine_fraction * delta
        n_fine = int(np.ceil(2.0 * step / fine_step)) + 1
        offsets = np.linspace(0.0, 2.0 * step, n_fine)
        fine = (grid[idx - 1][:, None] + offsets[None, :]).ravel()
        zf = np.asarray(ev(fine)).reshape(idx.size, n_fine)
        miss_rows = []
        lo_list, hi_list = [], []
        for row in range(idx.size):
            zr = zf[row]
            fl = np.nonzero(np.signbit(zr[:-1]) != np.signbit(zr[1:]))[0]
            base = grid[idx[row] - 1]
            if fl.size:
                lo_list.append(base + offsets[fl])
                hi_list.append(base + offsets[fl + 1])
            else:
                j = int(np.argmin(np.abs(zr)))
                miss_rows.append((base + offsets[j], float(np.abs(zr[j]))))
        if lo_list:
            rescued = _bisect_sign_changes(
                ev,
                np.concatenate(lo_list),
                np.concatenate(hi_list),
                iters=refine_iters,
            )
            zeros = np.concatenate([zeros, rescued])
        if miss_rows:
            near_misses = np.array(miss_rows, dtype=np.float64)

    zeros = np.unique(zeros[(zeros > t_lo) & (zeros < t_hi)])
    expected = float((riemann_siegel.theta(t_hi) - riemann_siegel.theta(t_lo)) / np.pi)
    return CensusWindow(
        t_lo=float(t_lo),
        t_hi=float(t_hi),
        zeros=zeros,
        expected_count=expected,
        rms_z=rms,
        step=step,
        rescued=np.sort(rescued),
        near_misses=near_misses,
    )


def normalized_gaps(zeros: np.ndarray) -> np.ndarray:
    """Unit-mean consecutive gaps: ``diff`` of the smooth-count unfolded zeros."""
    return np.diff(smooth_count(np.asarray(zeros, dtype=np.float64)))


def gue_small_gap_density(s: np.ndarray | float) -> np.ndarray:
    """Leading GUE level-repulsion law ``p(s) ~ (pi^2/3) s^2`` as ``s -> 0``.

    The exact sine-kernel coefficient -- the target of the small-gap tail read.
    (The Wigner surmise's small-``s`` coefficient is ``32/pi^2 ~ 3.24`` against
    the exact ``pi^2/3 ~ 3.29``; both are plotted by the script.)
    """
    s = np.asarray(s, dtype=np.float64)
    return (np.pi**2 / 3.0) * s**2


def gue_small_gap_cdf(s: np.ndarray | float) -> np.ndarray:
    """Cumulative small-gap law ``P(gap < s) ~ (pi^2/9) s^3`` -- the census target.

    The cumulative form is the statistically stable readout for a tail census
    (each bin of the density gets only a handful of events; the cumulative
    count uses them all).
    """
    s = np.asarray(s, dtype=np.float64)
    return (np.pi**2 / 9.0) * s**3


def _density_integral(a: float, b: float, c: float) -> float:
    """``int_a^b rho(x) / (x - c)^2 dx`` for the smooth density ``rho`` -- closed form.

    Antiderivative ``F(x) = [-log(x/2pi)/(x - c) + (1/c) log(|x - c|/x)] / 2pi``
    (differentiate to check; the absolute value covers both sides of ``c``),
    with ``F -> 0`` as ``x -> inf``. Requires ``c`` outside ``[a, b]``; used for
    the out-of-window tails of the CSV sum ``g`` (including the mirror axis,
    where ``c = -gamma < 0``).
    """

    def f(x: float) -> float:
        return (-np.log(x / TWO_PI) / (x - c) + np.log(abs(x - c) / x) / c) / TWO_PI

    if not a < b:
        return 0.0
    if a < c < b:
        raise ValueError("density tail requires c outside [a, b]")
    upper = 0.0 if np.isinf(b) else f(b)
    return float(upper - f(a))


def csv_g(
    zeros: np.ndarray,
    pair_index: int,
    *,
    t_lo: float | None = None,
    t_hi: float | None = None,
    neighborhood: int | None = None,
) -> float:
    """The CSV sum ``g`` for the consecutive pair ``zeros[i], zeros[i+1]``.

    Sums ``1/(gamma - g-)^2 + 1/(gamma - g+)^2`` over the censused zeros, then
    completes the three unseen regions -- below the window (down to ``2 pi``,
    where the smooth density vanishes), above it, and the entire mirror axis
    ``gamma < 0`` -- with the closed-form density tails. The window bounds
    default to the censused span itself.

    ``neighborhood`` truncates the explicit sum to that many censused zeros on
    each side of the pair (the density tails then start at the truncation
    edges), turning the per-pair cost from O(window) into O(neighborhood); at
    2000 neighbours the swapped-in density beyond contributes ~1e-4 of ``g``,
    far below the ``lambda`` sensitivity to ``g`` (second order for any
    high-quality pair).
    """
    x = np.asarray(zeros, dtype=np.float64)
    i = int(pair_index)
    gm, gp = float(x[i]), float(x[i + 1])
    lo = float(t_lo) if t_lo is not None else float(x[0])
    hi = float(t_hi) if t_hi is not None else float(x[-1])
    if neighborhood is not None:
        j0, j1 = max(0, i - neighborhood), min(x.size, i + 2 + neighborhood)
        if j0 > 0:
            lo = float(x[j0])
        if j1 < x.size:
            hi = float(x[j1 - 1])
        x = x[j0:j1]
        i -= j0

    others = np.delete(x, [i, i + 1])
    g = float(np.sum(1.0 / (others - gm) ** 2 + 1.0 / (others - gp) ** 2))
    for c in (gm, gp):
        g += _density_integral(TWO_PI, lo, c)  # below the window
        g += _density_integral(hi, np.inf, c)  # above the window
        g += _density_integral(TWO_PI, np.inf, -c)  # mirror zeros at -gamma
    return g


def csv_lambda(delta: float, g: float) -> float:
    """CSV lower bound ``lambda <= Lambda`` for a Lehmer pair, *classical* units.

    ``lambda_gamma = ((1 - 5 Delta^2 g / 4)^{4/5} - 1) / (8 g)`` in the
    gamma-coordinate flow (Stopple eq. 4), times 4 for the classical ``H_t`` /
    Polymath normalization all published ``Lambda`` bounds use (zeros at
    ``2 gamma``: ``Delta`` doubles, ``g`` quarters, ``Delta^2 g`` invariant).
    The factor is pinned empirically, not just derived: for a high-quality pair
    this is ``lambda ~ -Delta^2 / 2 + O(Delta^4 g)`` -- nearly independent of
    ``g`` -- and the COSV 1993 pair (``Delta ~ 1.0857e-4`` at ``t ~ 3.8886e8``,
    reconstructed from Stopple's section-6 data) gives ``-Delta^2/2 =
    -5.894e-9`` against the published bound ``-5.895e-9``. Requires
    ``Delta^2 g < 4/5`` (a Lehmer pair); raises otherwise.
    """
    d2g = delta * delta * g
    if not d2g < CSV_THRESHOLD:
        raise ValueError("not a Lehmer pair: Delta^2 g >= 4/5")
    return float(4.0 * ((1.0 - 1.25 * d2g) ** 0.8 - 1.0) / (8.0 * g))


@dataclass
class PairRow:
    """One censused consecutive pair and its CSV readouts."""

    gamma_minus: float
    gamma_plus: float
    s: float  # normalized gap
    delta2g: float  # CSV quality (Lehmer pair iff < 4/5)
    lam: float | None  # classical-normalization lambda <= Lambda (None if not Lehmer)
    below_floor: bool  # s under the fp64 resolution floor: needs mpmath polish


def lehmer_census(
    window: CensusWindow,
    *,
    s_cut: float = 0.8,
    quality_cut: float | None = None,
    neighborhood: int | None = 2000,
) -> list[PairRow]:
    """CSV census of the window: every consecutive pair with normalized gap < cut.

    ``s_cut`` pre-filters by normalized gap (measured ``Delta^2 g`` runs at
    ``~1.5-4 s^2`` for GUE-typical surroundings, so pairs past ``s ~ 0.8``
    cannot reach the ``4/5`` bar and the O(neighborhood) sum ``g`` is
    skipped). Rows are sorted by ascending ``Delta^2 g`` (best quality first);
    ``quality_cut`` optionally drops rows above a ``Delta^2 g`` value. Rows
    with ``s`` under the window's fp64 resolution floor
    (:func:`gap_resolution_floor` at the measured RMS amplitude) are flagged
    ``below_floor`` -- their gap is not trustworthy at fp64 and should be
    re-resolved with :func:`polish_pair` / :func:`resolve_near_miss` before
    quoting a ``lambda``.
    """
    x = window.zeros
    if x.size < 3:
        return []
    floor = float(
        gap_resolution_floor(0.5 * (window.t_lo + window.t_hi), z_scale=window.rms_z)
    )
    s_all = normalized_gaps(x)
    rows: list[PairRow] = []
    for i in np.nonzero(s_all < s_cut)[0]:
        g = csv_g(
            x, int(i), t_lo=window.t_lo, t_hi=window.t_hi, neighborhood=neighborhood
        )
        delta = float(x[i + 1] - x[i])
        d2g = delta * delta * g
        if quality_cut is not None and d2g >= quality_cut:
            continue
        lam = csv_lambda(delta, g) if d2g < CSV_THRESHOLD else None
        rows.append(
            PairRow(
                gamma_minus=float(x[i]),
                gamma_plus=float(x[i + 1]),
                s=float(s_all[i]),
                delta2g=float(d2g),
                lam=lam,
                below_floor=bool(s_all[i] < floor),
            )
        )
    rows.sort(key=lambda r: r.delta2g)
    return rows


def polish_pair(
    gamma_minus: float, gamma_plus: float, *, dps: int = 40
) -> tuple[float, float]:
    """Re-locate a censused pair with ``mpmath.siegelz`` at ``dps`` digits.

    The near-degenerate pairs are exactly where the fp64 evaluator is weakest
    (small ``|Z|``, cancellation), so the shortlist gets an arbitrary-precision
    polish: each zero is re-bracketed around its fp64 estimate and bisected on
    the sign of ``siegelz``. Validation tooling -- the censused values stay the
    forward product; this only measures how far fp64 moved them.
    """
    import mpmath as mp

    half = 0.25 * (gamma_plus - gamma_minus)
    out = []
    with mp.workdps(dps):
        for est in (gamma_minus, gamma_plus):
            lo, hi = mp.mpf(est) - half, mp.mpf(est) + half
            z_lo, z_hi = mp.siegelz(lo), mp.siegelz(hi)
            if mp.sign(z_lo) == mp.sign(z_hi):  # pragma: no cover - bad bracket
                raise ValueError("no sign change in polish bracket")
            for _ in range(60):  # interval / 2^60: far below fp64 resolution
                mid = (lo + hi) / 2
                z_mid = mp.siegelz(mid)
                if mp.sign(z_mid) == mp.sign(z_lo):
                    lo, z_lo = mid, z_mid
                else:
                    hi, z_hi = mid, z_mid
            out.append(float((lo + hi) / 2))
    return out[0], out[1]


def resolve_near_miss(
    t_center: float, *, span: float, n_grid: int = 101, dps: int = 30
) -> tuple[float, float] | None:
    """Resolve a sub-floor ``|Z|`` dip with ``mpmath``: a hidden pair, or not.

    A dip flagged by :func:`scan_zeros` whose interior extremum sits below the
    fp64 noise (``near_misses``, or a ``below_floor`` census row) cannot be
    trusted at fp64 -- the pair may be real, unresolved, or an artifact of the
    phase noise. This evaluates ``mpmath.siegelz`` on a fine grid of width
    ``span`` around the dip and bisects any pair of sign changes found,
    returning the resolved ``(gamma-, gamma+)`` or ``None`` if ``Z`` provably
    keeps its sign on the grid resolution. Pure validation tooling for the
    shortlisted candidates (each call is ~``n_grid + 120`` arbitrary-precision
    ``Z`` evaluations -- expensive; the shortlist is tiny).
    """
    import mpmath as mp

    with mp.workdps(dps):
        ts = [
            mp.mpf(t_center) - span / 2 + span * k / (n_grid - 1) for k in range(n_grid)
        ]
        zs = [mp.siegelz(t) for t in ts]
        flips = [k for k in range(n_grid - 1) if mp.sign(zs[k]) != mp.sign(zs[k + 1])]
        if len(flips) < 2:
            return None
        located = []
        for k in (flips[0], flips[-1]):
            lo, hi, z_lo = ts[k], ts[k + 1], zs[k]
            for _ in range(60):
                mid = (lo + hi) / 2
                z_mid = mp.siegelz(mid)
                if mp.sign(z_mid) == mp.sign(z_lo):
                    lo, z_lo = mid, z_mid
                else:
                    hi = mid
            located.append(float((lo + hi) / 2))
    return located[0], located[1]
