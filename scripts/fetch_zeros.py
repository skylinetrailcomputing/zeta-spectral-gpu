"""Fetch and summarise a large precomputed Riemann-zero set (Odlyzko tables).

Downloads + caches the table under data/ (gitignored) and prints a few sanity
stats. This is the ingestion foundation (issue #4) the spacing/pair-correlation
experiments build on; the zeros are only ever characterised here, never fitted.

    uv run python scripts/fetch_zeros.py                 # first 100k (zeros1)
    uv run python scripts/fetch_zeros.py --source first-2M --n 1000000
"""

from __future__ import annotations

import argparse

import numpy as np

from zeta_spectral_gpu import zeros


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        default="first-100k",
        choices=sorted(zeros.ODLYZKO_TABLES),
        help="which Odlyzko table to load",
    )
    ap.add_argument(
        "--n", type=int, default=None, help="how many ordinates (default all)"
    )
    args = ap.parse_args()

    table = zeros.ODLYZKO_TABLES[args.source]
    print(f"source={args.source}  url={table.url}")
    tau = zeros.load_odlyzko_ordinates(args.n, name=args.source)
    unfolded = zeros.unfold(tau)
    spacings = np.diff(unfolded)

    print(f"loaded {tau.size:,} ordinates  (accuracy ~{table.accuracy:g})")
    print(f"  tau range:        [{tau[0]:.6f}, {tau[-1]:.6f}]")
    print(f"  ascending:        {bool(np.all(np.diff(tau) > 0))}")
    print(f"  mean unfolded gap: {spacings.mean():.6f}  (should be ~1.0)")


if __name__ == "__main__":
    main()
