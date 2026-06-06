"""Reusable Dirichlet-``L`` zero-locator core (issue #60).

Packages the prime-driven mirror locator (#25/#42) as a small, reusable
scan -> peaks -> score pipeline so the CLI, a notebook, and any future web demo
share one code path instead of re-implementing the inline logic that used to live
in ``scripts/run_dirichlet_mirror.py``.

The forward object is unchanged (Sierra arXiv:1404.4252 eq. 13.6): the truncated
partial sum of ``1 / L(s, chi)``,

    M'_z(n) = sum_{k<=n} chi(k) mu(k) k^{-z},   z = 1/2 + iE,

whose magnitude ``|M'_z(E)|`` peaks at the ordinates of the zeros of ``L(s, chi)``.
The character ``chi`` (its modulus is the only number-theoretic input) goes in and
the zeros come out. The independently-computed ``L``-zeros
(:func:`dirichlet.lfunction_zeros`, mpmath) are used **only** to mark and score the
located peaks — they are never fed into the locator. That is the forward/inverse
line the project rule turns on (see the README and the issue #25 ruling); every
function here keeps the located peaks (forward) and the comparison zeros (scoring)
strictly separate.

Pieces:

* :func:`scan_locator` — run the scan on the GPU (NVRTC ``RawModule``, no cuSOLVER
  needed) with a transparent CPU fallback; returns the complex ``M'_z`` and the
  backend used.
* :func:`score_scan` — turn an already-computed ``|M'_z|`` into located peaks,
  comparison zeros, and the matched table.
* :func:`locate_and_score` — the one-call entry point (scan then score) used by the
  visualizer; returns a :class:`LocatorScan`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dirac_mirror as dm
from .dirichlet import is_real_character, lfunction_weights, lfunction_zeros


@dataclass(frozen=True)
class ZeroMatch:
    """One independent ``L``-zero paired with its nearest located peak."""

    true_E: float
    peak_E: float | None  # nearest located peak within tolerance, else None
    error: float | None  # |peak_E - true_E|, else None

    @property
    def matched(self) -> bool:
        return self.peak_E is not None


@dataclass(frozen=True, eq=False)
class LocatorScan:
    """Result of a forward locator scan scored against the independent zeros.

    ``peaks`` are the forward output (located from the character alone);
    ``true_zeros`` are the independent mpmath comparison points; ``matches`` pairs
    each true zero with its nearest peak. ``backend`` is the scan device
    (``"gpu"`` / ``"cpu"``) and ``is_real`` records which kernel/real-vs-complex
    path the character took.
    """

    grid: np.ndarray
    abs_m: np.ndarray
    peaks: np.ndarray
    true_zeros: np.ndarray
    matches: tuple[ZeroMatch, ...]
    height: float
    backend: str = "cpu"
    is_real: bool = True

    @property
    def matched_count(self) -> int:
        return sum(m.matched for m in self.matches)

    @property
    def kernel(self) -> str:
        """The GPU kernel the character's coefficients select (real vs complex)."""
        return "mobius_locator" if self.is_real else "weighted_locator"


def peak_threshold(n: int) -> float:
    """Height a ``|M'_z|`` local maximum must clear to count as a located peak.

    ``0.3 log n``: the forward growth at a zero is ``|M'_z(n)| ~ log n / |Z'(E)|``
    (Sierra eq. 12.30), so the threshold tracks the truncation; the ``0.3`` factor
    keeps the off-zero oscillations (bounded, ``O(1)``) below the bar.
    """
    return 0.3 * float(np.log(n))


def local_maxima(abs_m: np.ndarray, height: float) -> np.ndarray:
    """Indices of interior local maxima of ``abs_m`` strictly above ``height``."""
    abs_m = np.asarray(abs_m, dtype=np.float64)
    interior = (
        (abs_m[1:-1] > abs_m[:-2]) & (abs_m[1:-1] >= abs_m[2:]) & (abs_m[1:-1] > height)
    )
    return np.nonzero(interior)[0] + 1


def match_peaks(
    peaks: np.ndarray, true_zeros: np.ndarray, *, tol: float = 0.3
) -> tuple[ZeroMatch, ...]:
    """Pair each independent zero with its nearest located peak within ``tol``."""
    peaks = np.asarray(peaks, dtype=np.float64)
    matches: list[ZeroMatch] = []
    for g in np.asarray(true_zeros, dtype=np.float64):
        g = float(g)
        if peaks.size:
            j = int(np.argmin(np.abs(peaks - g)))
            err = abs(float(peaks[j]) - g)
            if err < tol:
                matches.append(ZeroMatch(g, float(peaks[j]), err))
                continue
        matches.append(ZeroMatch(g, None, None))
    return tuple(matches)


def scan_locator(
    char: np.ndarray,
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
    prefer_gpu: bool = True,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Scan ``M'_z(E)`` over ``grid`` for character ``char`` (forward, Piece A).

    Builds the ``chi*mu`` reflection coefficients (:func:`dirichlet.lfunction_weights`,
    or ``weights`` if supplied) and evaluates the truncated locator sum at every
    grid point. Tries the GPU kernel first (:func:`dirac_mirror_gpu.mobius_partial_sum_gpu`
    — NVRTC, so no cuSOLVER DLL wiring) and falls back to the CPU reference if CuPy
    is unavailable. Returns ``(M'_z, backend)`` with ``M'_z`` complex and ``backend``
    one of ``"gpu"`` / ``"cpu"``. No zeros enter — they are read off ``|M'_z|``
    downstream.
    """
    grid = np.asarray(grid, dtype=np.float64)
    w = lfunction_weights(char, n) if weights is None else weights
    if prefer_gpu:
        try:
            from . import dirac_mirror_gpu as gpu

            values = gpu.mobius_partial_sum_gpu(grid, n, sigma=sigma, weights=w)
            return np.asarray(values), "gpu"
        except ImportError:
            pass  # cupy missing -> CPU reference below
    values = dm.mobius_partial_sum(grid, n, sigma=sigma, weights=w)
    return np.asarray(values), "cpu"


def score_scan(
    grid: np.ndarray,
    abs_m: np.ndarray,
    *,
    height: float,
    char: np.ndarray | None = None,
    true_zeros: np.ndarray | None = None,
    match_tol: float = 0.3,
    e_min: float | None = None,
    e_max: float | None = None,
    backend: str = "cpu",
) -> LocatorScan:
    """Locate peaks in a precomputed ``|M'_z|`` and score them against the zeros.

    Splits scoring from scanning so a caller that already holds ``abs_m`` (e.g. the
    CLI, which times its own GPU/CPU scans) reuses the peak-find / match logic
    without rescanning. ``true_zeros`` are taken as given, or computed independently
    from ``char`` via :func:`dirichlet.lfunction_zeros` over ``[e_min, e_max]``
    (defaulting to the grid span). Exactly one of ``char`` / ``true_zeros`` is
    required. The comparison zeros are used only to score the peaks.
    """
    grid = np.asarray(grid, dtype=np.float64)
    abs_m = np.asarray(abs_m, dtype=np.float64)
    peaks = grid[local_maxima(abs_m, height)]
    if true_zeros is None:
        if char is None:
            raise ValueError("pass either true_zeros or char to compute them")
        lo = float(grid[0]) if e_min is None else e_min
        hi = float(grid[-1]) if e_max is None else e_max
        true_zeros = lfunction_zeros(char, e_min=lo, e_max=hi)
    true_zeros = np.asarray(true_zeros, dtype=np.float64)
    matches = match_peaks(peaks, true_zeros, tol=match_tol)
    is_real = True if char is None else is_real_character(char)
    return LocatorScan(
        grid=grid,
        abs_m=abs_m,
        peaks=peaks,
        true_zeros=true_zeros,
        matches=matches,
        height=height,
        backend=backend,
        is_real=is_real,
    )


def locate_and_score(
    char: np.ndarray,
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
    prefer_gpu: bool = True,
    height: float | None = None,
    match_tol: float = 0.3,
    weights: np.ndarray | None = None,
    true_zeros: np.ndarray | None = None,
    e_min: float | None = None,
    e_max: float | None = None,
) -> LocatorScan:
    """Scan then score in one call — the visualizer's entry point.

    Runs :func:`scan_locator` (GPU with CPU fallback) and feeds ``|M'_z|`` to
    :func:`score_scan`. ``height`` defaults to :func:`peak_threshold` of ``n``.
    Returns a :class:`LocatorScan` carrying the grid, ``|M'_z|``, located peaks,
    independent zeros, and the matched table — everything the figure and the CLI
    table need. Forward: only ``char`` (its modulus) drives the location; the zeros
    are computed solely to score.
    """
    grid = np.asarray(grid, dtype=np.float64)
    values, backend = scan_locator(
        char, n, grid, sigma=sigma, prefer_gpu=prefer_gpu, weights=weights
    )
    h = peak_threshold(n) if height is None else height
    return score_scan(
        grid,
        np.abs(values),
        height=h,
        char=char,
        true_zeros=true_zeros,
        match_tol=match_tol,
        e_min=e_min,
        e_max=e_max,
        backend=backend,
    )
