"""CPU reference statistics — the ground truth the GPU paths must reproduce.

Everything here is plain numpy/scipy and deliberately simple. The GPU versions
in ``spacing_gpu.py`` exist for *scale*, not to change answers; the test suite
asserts GPU-vs-CPU agreement on small inputs against these functions.
"""

from __future__ import annotations

import numpy as np

PI = np.pi


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
