"""CPU reference statistics — the ground truth the GPU paths must reproduce.

Everything here is plain numpy/scipy and deliberately simple. The GPU versions
in ``spacing_gpu.py`` exist for *scale*, not to change answers; the test suite
asserts GPU-vs-CPU agreement on small inputs against these functions.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici

PI = np.pi
EULER_GAMMA = np.euler_gamma


def nearest_neighbour_spacings(unfolded: np.ndarray) -> np.ndarray:
    """Consecutive differences of an ascending unfolded level sequence."""
    x = np.asarray(unfolded, dtype=np.float64)
    return np.diff(x)


def gue_wigner_surmise(s: np.ndarray | float) -> np.ndarray:
    """GUE (unitary) nearest-neighbour surmise: p(s) = 32/pi^2 s^2 e^{-4 s^2/pi}."""
    s = np.asarray(s, dtype=np.float64)
    return (32.0 / PI**2) * s**2 * np.exp(-4.0 * s**2 / PI)


def goe_wigner_surmise(s: np.ndarray | float) -> np.ndarray:
    """GOE (orthogonal) surmise: p(s) = (pi/2) s e^{-pi s^2/4}."""
    s = np.asarray(s, dtype=np.float64)
    return (PI / 2.0) * s * np.exp(-PI * s**2 / 4.0)


def poisson_surmise(s: np.ndarray | float) -> np.ndarray:
    """Poisson (uncorrelated) surmise: p(s) = e^{-s}."""
    s = np.asarray(s, dtype=np.float64)
    return np.exp(-s)


def montgomery_pair_correlation(r: np.ndarray | float) -> np.ndarray:
    """Montgomery's pair-correlation form R2(r) = 1 - (sin(pi r) / (pi r))^2.

    The GUE sine-kernel two-point function: 0 at r=0 (level repulsion), → 1 as
    r → ∞ (decorrelation). ``np.sinc(r)`` is exactly ``sin(pi r)/(pi r)``.
    """
    r = np.asarray(r, dtype=np.float64)
    return 1.0 - np.sinc(r) ** 2


def spacing_density(
    spacings: np.ndarray, n_bins: int = 50, s_max: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram spacings into a probability density. Returns (centres, density)."""
    s = np.asarray(spacings, dtype=np.float64)
    counts, edges = np.histogram(s, bins=n_bins, range=(0.0, s_max), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def pair_correlation_histogram(
    unfolded: np.ndarray, bin_width: float, max_sep: float
) -> np.ndarray:
    """Histogram of all pair separations ``|x_i - x_j|`` up to ``max_sep``.

    O(N^2) reference. Because the levels are ascending we only look forward and
    break out of the inner scan once the separation exceeds ``max_sep`` — which
    is exactly the structure the GPU kernel mirrors.
    """
    x = np.sort(np.asarray(unfolded, dtype=np.float64))
    n_bins = int(np.ceil(max_sep / bin_width))
    hist = np.zeros(n_bins, dtype=np.int64)
    n = x.size
    for i in range(n):
        d = x[i + 1 :] - x[i]
        d = d[d < max_sep]
        if d.size:
            bins = (d / bin_width).astype(np.int64)
            bins = bins[bins < n_bins]
            np.add.at(hist, bins, 1)
    return hist


def pair_correlation_density(
    hist: np.ndarray, bin_width: float, n_levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a forward pair-separation histogram to an empirical R2(r).

    ``hist[b]`` counts ordered pairs (i<j) with separation in bin ``b``. For
    unfolded levels (unit density), the expected forward count in a width-``w``
    bin at separation ``r`` is ``n_levels * R2(r) * w`` (edge effects O(max_sep/N),
    negligible at scale), so ``R2(r) = hist / (n_levels * w)``. Returns
    ``(r_centres, R2)`` to compare against ``montgomery_pair_correlation``.
    """
    hist = np.asarray(hist, dtype=np.float64)
    centres = (np.arange(hist.size) + 0.5) * bin_width
    r2 = hist / (n_levels * bin_width)
    return centres, r2


# --- Long-range rigidity: number variance and the Dyson-Mehta Delta_3 -------
#
# Where the nearest-neighbour spacing and pair correlation probe local (one- and
# two-level) structure, Sigma^2(L) and Delta_3(L) measure how *rigid* the spectrum
# is over a window of L mean spacings. The discriminating signature is the growth
# rate: GUE grows only logarithmically in L (a rigid spectrum), Poisson linearly.


def number_variance(
    unfolded: np.ndarray, lengths: np.ndarray | float, *, n_offsets: int = 2000
) -> np.ndarray:
    """Number variance ``Sigma^2(L)`` for each window length in ``lengths``.

    For each ``L`` we slide ``n_offsets`` evenly-spaced windows across the
    unfolded sequence, count the levels in each (``searchsorted`` gives the
    half-open count ``#{x in [s, s + L)}``), and take the variance of those
    counts. GUE grows logarithmically with ``L`` (rigid); Poisson grows like
    ``L``. ``unfolded`` need not be pre-sorted. Returns one ``Sigma^2`` per
    requested ``L`` (NaN where ``L`` exceeds the available span).
    """
    x = np.sort(np.asarray(unfolded, dtype=np.float64))
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    span = x[-1] - x[0]
    out = np.full(lengths.shape, np.nan)
    for i, length in enumerate(lengths):
        if not (0.0 < length < span):
            continue
        starts = np.linspace(x[0], x[-1] - length, n_offsets)
        counts = np.searchsorted(x, starts + length, side="left") - np.searchsorted(
            x, starts, side="left"
        )
        out[i] = counts.var()
    return out


def dyson_mehta_delta3(
    unfolded: np.ndarray,
    lengths: np.ndarray | float,
    *,
    n_offsets: int = 2000,
    n_grid: int = 400,
) -> np.ndarray:
    """Dyson-Mehta spectral rigidity ``Delta_3(L)`` for each window length.

    ``Delta_3(L) = (1/L) min_{a,b} integral_0^L [N(x) - a - b x]^2 dx``, averaged
    over window starts. We obtain it via the Mehta transform of the empirical
    number variance (:func:`delta3_from_sigma2` applied to :func:`number_variance`
    on a fine ``r`` grid) rather than a direct per-window staircase line fit.

    The reason is precision: a direct fit needs the first and second moments of the
    level positions within each window, and once the unfolded ordinates exceed
    ~1e6 the global prefix sums used to form them (cumulative ``x`` and ``x^2``)
    overflow fp64's exact-integer range, so differencing them loses ~10 digits.
    The transform only ever counts levels (integers), so it stays exact at scale.
    GUE grows like ``(1/2 pi^2) ln L`` (rigid); Poisson like ``L/15``.
    """
    x = np.sort(np.asarray(unfolded, dtype=np.float64))
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    span = x[-1] - x[0]
    out = np.full(lengths.shape, np.nan)
    valid = (lengths > 0.0) & (lengths < span)
    if not valid.any():
        return out
    r = np.linspace(0.0, float(lengths[valid].max()), n_grid)
    s2 = np.empty_like(r)
    s2[0] = 0.0  # Sigma^2(0) = 0
    s2[1:] = number_variance(x, r[1:], n_offsets=n_offsets)
    out[valid] = delta3_from_sigma2(
        lambda q: np.interp(q, r, s2), lengths[valid], n_grid=n_grid
    )
    return out


def gue_number_variance(length: np.ndarray | float) -> np.ndarray:
    """Exact GUE number variance from the sine kernel.

        Sigma^2(L) = (1/pi^2)[ln(2 pi L) + gamma + 1 - cos(2 pi L) - Ci(2 pi L)]
                     + L [1 - (2/pi) Si(2 pi L)].

    Reduces to ``Sigma^2 -> L`` as ``L -> 0`` (small windows hold at most one
    level) and to the logarithmic asymptote ``(1/pi^2)(ln 2 pi L + gamma + 1)``
    as ``L -> infinity``. ``Si``/``Ci`` come from ``scipy.special.sici``.
    """
    length = np.asarray(length, dtype=np.float64)
    z = 2.0 * PI * length
    with np.errstate(divide="ignore", invalid="ignore"):
        si, ci = sici(z)
        value = (np.log(z) + EULER_GAMMA + 1.0 - np.cos(z) - ci) / PI**2 + length * (
            1.0 - (2.0 / PI) * si
        )
    return np.where(length > 0.0, value, 0.0)


def delta3_from_sigma2(
    sigma2, lengths: np.ndarray | float, *, n_grid: int = 2000
) -> np.ndarray:
    """``Delta_3(L)`` from a number variance via the Mehta integral transform.

        Delta_3(L) = (2 / L^4) integral_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr

    (Mehta, *Random Matrices*, ch. 16). ``sigma2`` is any callable ``r ->
    Sigma^2(r)`` — pass :func:`gue_number_variance` for the GUE reference curve,
    or an interpolant of an empirical ``Sigma^2``. The kernel is verified by the
    Poisson case ``Sigma^2(r) = r``, which gives ``Delta_3(L) = L/15``.
    """
    arr = np.asarray(lengths, dtype=np.float64)
    flat = np.atleast_1d(arr)
    out = np.empty(flat.shape)
    for i, length in enumerate(flat):
        r = np.linspace(0.0, length, n_grid)
        s2 = np.asarray(sigma2(r), dtype=np.float64)
        kernel = length**3 - 2.0 * length**2 * r + r**3
        out[i] = (2.0 / length**4) * np.trapezoid(kernel * s2, r)
    return out if arr.ndim else out[0]


def gue_delta3(length: np.ndarray | float) -> np.ndarray:
    """Exact GUE ``Delta_3(L)``: the Mehta transform of :func:`gue_number_variance`.

    Large-``L`` asymptote ``(1/2 pi^2)(ln 2 pi L + gamma - 5/4)`` — half the slope
    of ``Sigma^2`` in ``ln L``, the textbook signature of a rigid GUE spectrum.
    """
    return delta3_from_sigma2(gue_number_variance, length)
