"""Warm-up #35: the spacing-ratio statistic r̃_n of the Riemann zeros vs GUE.

The consecutive spacing ratio r̃_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) (Atas,
Bogomolny, Giraud & Roux 2013) is the *unfolding-free* universality readout: it is
a ratio of adjacent **raw** gaps, so the local mean density cancels — no Odlyzko
unfold, no fitted smooth count. That makes it the robust discriminator for every
small-N forward spectrum in this repo, where the unfolding that P(s)/Σ²/Δ₃ need is
the weak link (see #9, #20).

Loads raw zero ordinates (NOT unfolded), computes r̃ on the GPU with a CPU
cross-check, prints ⟨r̃⟩ against the GUE/GOE/Poisson surmise means, and saves the
P(r̃) histogram figure. Forward check: the zeros are only an *output* being
characterised — nothing is fitted to them.

    uv run python scripts/run_spacing_ratios.py                  # 100k zeros, GPU, figure
    uv run python scripts/run_spacing_ratios.py --source first-2M
    uv run python scripts/run_spacing_ratios.py --source mpmath --n 2000 --no-gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _ratios(levels: np.ndarray, use_gpu: bool) -> tuple[np.ndarray, str]:
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            return spacing_gpu.spacing_ratios_gpu(levels), "gpu"
        except ImportError:
            return spacing.spacing_ratios(levels), "cpu (cupy unavailable)"
    return spacing.spacing_ratios(levels), "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        default="first-100k",
        help="zero source: 'mpmath' or an Odlyzko table (first-100k, first-2M)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="number of zeros (default: all; 2000 mpmath)",
    )
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument(
        "--out", type=Path, default=DATA / "gue_spacing_ratio.png", help="figure path"
    )
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU path")
    args = ap.parse_args()

    n = args.n
    if args.source == "mpmath" and n is None:
        n = 2000

    # NOTE: raw ordinates, deliberately NOT unfolded — r̃ is scale-free.
    tau = zeros.load_ordinates(n, source=args.source, cache_dir=DATA)
    rt, backend = _ratios(tau, use_gpu=not args.no_gpu)
    rt = rt[np.isfinite(rt)]
    mean_emp = float(np.mean(rt))

    print(f"source={args.source}  N={tau.size:,}  backend={backend}  (NO unfolding)")
    print(f"  <r~> = {mean_emp:.5f}")
    print(
        f"    GUE surmise     {spacing.MEAN_RATIO_GUE:.4f}   (delta {mean_emp - spacing.MEAN_RATIO_GUE:+.4f})"
    )
    print(f"    GOE surmise     {spacing.MEAN_RATIO_GOE:.4f}")
    print(f"    Poisson         {spacing.MEAN_RATIO_POISSON:.4f}")
    if backend == "gpu":
        cpu = spacing.spacing_ratios(tau)
        cpu = cpu[np.isfinite(cpu)]
        print(f"  GPU vs CPU max|delta| = {np.max(np.abs(cpu - rt)):.2e}")

    out = plots.spacing_ratio_figure(
        rt,
        out_path=args.out,
        n_bins=args.bins,
        title=f"Riemann-zero spacing ratio $\\tilde r$ vs GUE  (N={tau.size:,})",
    )
    print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
