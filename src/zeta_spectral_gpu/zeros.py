"""Riemann zero ordinates and spectral unfolding.

The nontrivial zeros are ``1/2 + i*tau_n`` with ``tau_n > 0``. We work with the
ordinates ``tau_n`` (generated to full precision via mpmath) and *unfold* them
so that the mean nearest-neighbour spacing is 1 — the prerequisite for comparing
local statistics against the random-matrix surmises.

This module only ever *produces* zeros to compare against; nothing here feeds
zeros into an operator construction. (Forward, not inverse — see README.)
"""

from __future__ import annotations

import csv
from pathlib import Path

import mpmath
import numpy as np

TWO_PI = 2.0 * np.pi


def riemann_zero_ordinates(
    n: int, cache_path: Path | None = None, dps: int = 30
) -> np.ndarray:
    """Return the first ``n`` positive ordinates ``tau_1..tau_n``.

    Uses ``mpmath.zetazero`` (Riemann–Siegel / Gram-point based) and caches the
    results as CSV so repeat runs are cheap. Returned as float64; the cache keeps
    full-precision strings for any higher-precision needs.
    """
    cached: list[float] = []
    if cache_path is not None and cache_path.exists():
        with cache_path.open(newline="") as fh:
            cached = [float(row["tau"]) for row in csv.DictReader(fh)]
    if len(cached) >= n:
        return np.asarray(cached[:n], dtype=np.float64)

    mpmath.mp.dps = dps
    rows: list[tuple[int, str]] = []
    taus: list[float] = []
    for k in range(1, n + 1):
        tau = mpmath.zetazero(k).imag
        taus.append(float(tau))
        rows.append((k, mpmath.nstr(tau, dps)))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "tau"])
            writer.writerows(rows)

    return np.asarray(taus, dtype=np.float64)


def smooth_count(t: np.ndarray | float) -> np.ndarray:
    """Smooth (average) zero-counting function ``N_bar(T)``.

    Leading Riemann–von Mangoldt asymptotic:
        N_bar(T) = (T / 2pi) * log(T / 2pi) - T / 2pi + 7/8.
    Good enough for unfolding so that spacings have unit mean.
    """
    t = np.asarray(t, dtype=np.float64)
    return t / TWO_PI * np.log(t / TWO_PI) - t / TWO_PI + 0.875


def unfold(ordinates: np.ndarray) -> np.ndarray:
    """Map ordinates ``tau_n`` to unfolded levels ``N_bar(tau_n)``.

    After unfolding, ``np.diff`` of the result has mean ≈ 1, which is what the
    GUE/GOE/Poisson surmises are normalised against.
    """
    return smooth_count(np.asarray(ordinates, dtype=np.float64))
