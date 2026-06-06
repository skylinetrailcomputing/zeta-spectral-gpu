"""Warm-up experiment #5: nearest-neighbour spacing of the Riemann zeros vs GUE.

Loads a large precomputed zero set (Odlyzko table, issue #4), unfolds it, computes
the nearest-neighbour spacings on the GPU (with a CPU cross-check), and:
  * prints a text summary + GPU-vs-CPU agreement,
  * saves a histogram figure vs the GUE Wigner surmise and Poisson,
  * optionally benchmarks the GPU vs CPU spacing reduction.

This is the headline warm-up result: the textbook Montgomery-Odlyzko GUE
agreement of the zero spacings, reproduced at scale. Forward check: the zeros are
only an *output* being characterised — nothing is fitted to them.

    uv run python scripts/run_spacing.py                       # 100k zeros, GPU, figure
    uv run python scripts/run_spacing.py --source first-2M --benchmark
    uv run python scripts/run_spacing.py --source mpmath --n 2000 --no-gpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _spacings(unfolded: np.ndarray, use_gpu: bool) -> tuple[np.ndarray, str]:
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            return spacing_gpu.nearest_neighbour_spacings_gpu(unfolded), "gpu"
        except ImportError:
            return spacing.nearest_neighbour_spacings(
                unfolded
            ), "cpu (cupy unavailable)"
    return spacing.nearest_neighbour_spacings(unfolded), "cpu"


def _benchmark(unfolded: np.ndarray, repeats: int) -> tuple[float, float, float]:
    """Mean ms/run for the CPU and GPU spacing reductions, plus max abs disagreement."""
    import cupy as cp

    from zeta_spectral_gpu import spacing_gpu

    t0 = time.perf_counter()
    for _ in range(repeats):
        cpu = spacing.nearest_neighbour_spacings(unfolded)
    cpu_ms = (time.perf_counter() - t0) / repeats * 1e3

    spacing_gpu.nearest_neighbour_spacings_gpu(unfolded)  # warm up JIT/alloc
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        gpu = spacing_gpu.nearest_neighbour_spacings_gpu(unfolded)
    cp.cuda.Device().synchronize()
    gpu_ms = (time.perf_counter() - t0) / repeats * 1e3

    return cpu_ms, gpu_ms, float(np.max(np.abs(cpu - gpu)))


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
        help="number of zeros (default: all; 2000 for mpmath)",
    )
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--s-max", type=float, default=4.0)
    ap.add_argument(
        "--out", type=Path, default=DATA / "gue_spacing.png", help="figure output path"
    )
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU path")
    ap.add_argument("--benchmark", action="store_true", help="time GPU vs CPU")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    n = args.n
    if args.source == "mpmath" and n is None:
        n = 2000  # mpmath is slow; a sane default

    tau = zeros.load_ordinates(n, source=args.source, cache_dir=DATA)
    unfolded = zeros.unfold(tau)
    spac, backend = _spacings(unfolded, use_gpu=not args.no_gpu)

    print(f"source={args.source}  N={tau.size:,}  backend={backend}")
    print(f"  tau range [{tau[0]:.3f}, {tau[-1]:.3f}]   mean spacing={spac.mean():.4f}")
    if backend == "gpu":
        cpu = spacing.nearest_neighbour_spacings(unfolded)
        print(f"  GPU vs CPU max|delta| = {np.max(np.abs(cpu - spac)):.2e}")

    out = plots.spacing_histogram_figure(
        spac,
        out_path=args.out,
        n_bins=args.bins,
        s_max=args.s_max,
        title=f"Riemann-zero spacing vs GUE  (source={args.source}, N={tau.size:,})",
    )
    print(f"  figure -> {out}")

    if args.benchmark:
        if backend != "gpu":
            print("  (benchmark needs the GPU path; skipped)")
        else:
            cpu_ms, gpu_ms, max_abs = _benchmark(unfolded, args.repeats)
            print(f"  benchmark over {args.repeats} runs (N={tau.size:,}):")
            print(f"    CPU np.diff      {cpu_ms:8.3f} ms")
            print(f"    GPU kernel       {gpu_ms:8.3f} ms   (x{cpu_ms / gpu_ms:.2f})")
            print(f"    agreement max|delta| {max_abs:.2e}")
            if cpu_ms / gpu_ms < 1.5:
                print(
                    "    note: NN spacing is O(N) and transfer-bound, so the kernel's\n"
                    "          real win is the O(N^2) pair-correlation scan (issue #6)."
                )


if __name__ == "__main__":
    main()
