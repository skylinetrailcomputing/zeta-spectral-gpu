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
import gzip
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import mpmath
import numpy as np

TWO_PI = 2.0 * np.pi

# Repo-root data dir (gitignored). zeros.py lives at src/zeta_spectral_gpu/.
DATA = Path(__file__).resolve().parents[2] / "data"


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


# --- Large precomputed zero sets (Odlyzko tables) ---------------------------
#
# mpmath.zetazero is precise but slow — fine to a few thousand, far too slow for
# the 10^6+ sets that make the spacing statistics crisp. Andrew Odlyzko's tables
# provide large precomputed ordinate lists; we cache them under data/ (gitignored)
# and parse them to the same float64 ndarray interface as riemann_zero_ordinates.
# Still forward: these are zeros we *characterise*, never fitted to anything.


@dataclass(frozen=True)
class OdlyzkoTable:
    """A downloadable Odlyzko zero table: one ascending ordinate per line."""

    name: str
    url: str
    count: int  # number of ordinates in the file
    accuracy: float  # absolute accuracy of each ordinate
    gzipped: bool = False


ODLYZKO_TABLES: dict[str, OdlyzkoTable] = {
    "first-100k": OdlyzkoTable(
        name="first-100k",
        url="https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1",
        count=100_000,
        accuracy=3e-9,
    ),
    "first-2M": OdlyzkoTable(
        name="first-2M",
        url="https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6.gz",
        count=2_001_052,
        accuracy=4e-9,
        gzipped=True,
    ),
}


def _odlyzko_cache_path(table: OdlyzkoTable, cache_dir: Path) -> Path:
    """Local (decompressed) cache file for a table."""
    return cache_dir / f"odlyzko_{table.name}.txt"


def parse_odlyzko(text: str | bytes) -> np.ndarray:
    """Parse Odlyzko table text (one ordinate per line) to a float64 array.

    The column is the imaginary part ``tau_n`` of the n-th zero, ascending. We
    split on all whitespace, which absorbs the files' fixed-width leading spaces.
    """
    if isinstance(text, bytes):
        text = text.decode("ascii")
    return np.asarray(text.split(), dtype=np.float64)


def download_odlyzko(
    name: str = "first-100k", cache_dir: Path = DATA, *, force: bool = False
) -> Path:
    """Download (and decompress) an Odlyzko table into ``cache_dir``; return path.

    A cheap no-op when already cached unless ``force``. Reads a public static
    file over HTTPS — no auth, and no zeros consumed as input (forward, not
    inverse).
    """
    table = ODLYZKO_TABLES[name]
    dest = _odlyzko_cache_path(table, cache_dir)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(table.url, timeout=120) as resp:
        raw = resp.read()
    dest.write_bytes(gzip.decompress(raw) if table.gzipped else raw)
    return dest


def load_odlyzko_ordinates(
    n: int | None = None,
    *,
    name: str = "first-100k",
    cache_dir: Path = DATA,
    download: bool = True,
) -> np.ndarray:
    """Return the first ``n`` ordinates (all if ``n`` is None) from a table.

    Downloads + caches on first use; ``download=False`` requires a pre-populated
    cache and raises otherwise. Same return type as ``riemann_zero_ordinates``,
    so ``unfold`` and the statistics consume it unchanged.
    """
    table = ODLYZKO_TABLES[name]
    dest = _odlyzko_cache_path(table, cache_dir)
    if not dest.exists():
        if not download:
            raise FileNotFoundError(
                f"no cached {name} table at {dest}; call with download=True"
            )
        dest = download_odlyzko(name, cache_dir)

    taus = parse_odlyzko(dest.read_bytes())
    if taus.size != table.count:
        warnings.warn(
            f"{name}: expected {table.count} ordinates, parsed {taus.size} "
            f"(possible partial download at {dest})",
            stacklevel=2,
        )
    if n is not None:
        if n > taus.size:
            raise ValueError(f"requested {n} ordinates but {name} has {taus.size}")
        taus = taus[:n]
    return taus


def load_ordinates(
    n: int | None = None, *, source: str = "mpmath", cache_dir: Path = DATA
) -> np.ndarray:
    """Unified ordinate accessor for the experiment scripts.

    ``source="mpmath"`` uses the precise-but-slow generator; any key in
    ``ODLYZKO_TABLES`` (e.g. ``"first-100k"``, ``"first-2M"``) uses the fast
    precomputed loader. ``n=None`` returns the whole Odlyzko table; the mpmath
    source requires an explicit ``n``. Default stays mpmath so nothing downloads
    by surprise.
    """
    if source == "mpmath":
        if n is None:
            raise ValueError("the 'mpmath' source requires an explicit n")
        return riemann_zero_ordinates(n, cache_path=cache_dir / "riemann_zeros.csv")
    if source in ODLYZKO_TABLES:
        return load_odlyzko_ordinates(n, name=source, cache_dir=cache_dir)
    raise ValueError(
        f"unknown source {source!r}; use 'mpmath' or one of {list(ODLYZKO_TABLES)}"
    )


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
