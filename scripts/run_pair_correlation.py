"""Warm-up experiment #6: pair correlation of the Riemann zeros vs the GUE sine kernel.

Loads a large precomputed zero set (Odlyzko table, issue #4), unfolds it, runs the
hand-written ``pair_correlation_hist`` CUDA kernel to histogram all forward pair
separations, normalises to an empirical R2(r), and:
  * cross-checks the GPU histogram against the CPU reference on a small subset,
  * saves R2(r) vs Montgomery's prediction 1 - (sin(pi r) / (pi r))^2,
  * optionally benchmarks the GPU vs CPU pair scan (the O(N^2) scan is where the
    kernel earns its keep).

Forward check: the zeros are only an *output* being characterised — nothing is
fitted to them.

    uv run python scripts/run_pair_correlation.py                   # 100k zeros, GPU
    uv run python scripts/run_pair_correlation.py --source first-2M --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _hist(unfolded, bin_width, max_sep, use_gpu):
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            return (
                spacing_gpu.pair_correlation_histogram_gpu(
                    unfolded, bin_width, max_sep
                ),
                "gpu",
            )
        except ImportError:
            pass
    return spacing.pair_correlation_histogram(unfolded, bin_width, max_sep), "cpu"


def _benchmark(unfolded, bin_width, max_sep, bench_n, repeats):
    import cupy as cp

    from zeta_spectral_gpu import spacing_gpu

    sub = unfolded[:bench_n]

    t0 = time.perf_counter()
    for _ in range(repeats):
        cpu = spacing.pair_correlation_histogram(sub, bin_width, max_sep)
    cpu_ms = (time.perf_counter() - t0) / repeats * 1e3

    spacing_gpu.pair_correlation_histogram_gpu(sub, bin_width, max_sep)  # warm up
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        gpu = spacing_gpu.pair_correlation_histogram_gpu(sub, bin_width, max_sep)
    cp.cuda.Device().synchronize()
    gpu_ms = (time.perf_counter() - t0) / repeats * 1e3

    agree = bool(np.array_equal(cpu, gpu))
    return cpu_ms, gpu_ms, agree


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source", default="first-100k", help="'mpmath' or an Odlyzko table"
    )
    ap.add_argument(
        "--n", type=int, default=None, help="number of zeros (default: all)"
    )
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--max-sep", type=float, default=3.0)
    ap.add_argument(
        "--out", type=Path, default=DATA / "pair_correlation.png", help="figure path"
    )
    ap.add_argument("--no-gpu", action="store_true", help="force the (slow) CPU path")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-n", type=int, default=20_000, help="N for the benchmark")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    n = args.n
    if args.source == "mpmath" and n is None:
        n = 2000

    tau = zeros.load_ordinates(n, source=args.source, cache_dir=DATA)
    unfolded = zeros.unfold(tau)
    hist, backend = _hist(
        unfolded, args.bin_width, args.max_sep, use_gpu=not args.no_gpu
    )

    print(f"source={args.source}  N={tau.size:,}  backend={backend}")
    print(f"  bin_width={args.bin_width}  max_sep={args.max_sep}  pairs={hist.sum():,}")

    if backend == "gpu":
        m = min(2000, tau.size)
        cpu_sub = spacing.pair_correlation_histogram(
            unfolded[:m], args.bin_width, args.max_sep
        )
        from zeta_spectral_gpu import spacing_gpu

        gpu_sub = spacing_gpu.pair_correlation_histogram_gpu(
            unfolded[:m], args.bin_width, args.max_sep
        )
        print(
            f"  GPU vs CPU agree on first {m}: {bool(np.array_equal(cpu_sub, gpu_sub))}"
        )

    out = plots.pair_correlation_figure(
        hist,
        args.bin_width,
        n_levels=tau.size,
        out_path=args.out,
        title=f"Zero pair correlation vs GUE sine kernel  (N={tau.size:,})",
    )
    print(f"  figure -> {out}")

    if args.benchmark:
        if backend != "gpu":
            print("  (benchmark needs the GPU path; skipped)")
        else:
            cpu_ms, gpu_ms, agree = _benchmark(
                unfolded, args.bin_width, args.max_sep, args.bench_n, args.repeats
            )
            print(f"  benchmark over {args.repeats} runs (N={args.bench_n:,}):")
            print(f"    CPU reference {cpu_ms:9.3f} ms")
            print(f"    GPU kernel    {gpu_ms:9.3f} ms   (x{cpu_ms / gpu_ms:.1f})")
            print(f"    agree: {agree}")


if __name__ == "__main__":
    main()
