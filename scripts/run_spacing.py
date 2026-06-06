"""Warm-up experiment: nearest-neighbour spacing of the Riemann zeros vs GUE.

Generates the first N zero ordinates, unfolds them, and compares the spacing
distribution against the GUE/GOE/Poisson surmises. Uses the GPU path when CuPy
is available, falling back to the CPU reference otherwise (the two must agree).

Forward check: the zeros are only ever an *output* being characterised here —
nothing is fitted to them.

Run (once the env is built):
    uv run python scripts/run_spacing.py --n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="number of zeros")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU path")
    args = ap.parse_args()

    tau = zeros.riemann_zero_ordinates(args.n, cache_path=DATA / "riemann_zeros.csv")
    unfolded = zeros.unfold(tau)

    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            s = spacing_gpu.nearest_neighbour_spacings_gpu(unfolded)
            backend = "gpu"
        except ImportError:
            s = spacing.nearest_neighbour_spacings(unfolded)
            backend = "cpu (cupy unavailable)"
    else:
        s = spacing.nearest_neighbour_spacings(unfolded)
        backend = "cpu"

    centres, density = spacing.spacing_density(s, n_bins=args.bins)
    print(f"backend={backend}  N={args.n}  mean spacing={np.mean(s):.4f}")
    print("  s      empirical   GUE       Poisson")
    for c, d in zip(centres, density):
        print(
            f"  {c:5.2f}  {d:9.4f}  {spacing.gue_wigner_surmise(c):8.4f}"
            f"  {spacing.poisson_surmise(c):8.4f}"
        )


if __name__ == "__main__":
    main()
