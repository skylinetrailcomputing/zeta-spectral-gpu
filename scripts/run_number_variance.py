"""Warm-up experiment #15: spectral rigidity of the Riemann zeros vs GUE.

Long-range companion to #5 (nearest-neighbour spacing) and #6 (pair correlation):
where those probe local structure, the number variance Sigma^2(L) and the
Dyson-Mehta Delta_3(L) probe how *rigid* the spectrum is over a window of L mean
spacings. GUE grows only logarithmically in L (rigid); Poisson grows linearly.

Loads a large precomputed zero set (Odlyzko table, issue #4), unfolds it, computes
Sigma^2(L) and Delta_3(L) on the GPU over a log-spaced grid of window lengths, and:
  * cross-checks the GPU statistics against the CPU reference on a small subset,
  * saves Sigma^2(L) and Delta_3(L) vs their GUE + Poisson reference curves,
  * optionally benchmarks the GPU vs CPU rigidity sweep.

At scale the zeros track GUE only up to the Berry saturation scale
``L* ~ ln(T/2pi) / pi`` (~3 for these heights); beyond it the arithmetic (prime)
contributions make the number variance saturate *below* the GUE logarithm — the
zeros are more rigid than GUE at long range (Berry, Nonlinearity 1, 1988). The
figure marks ``L*`` so the GUE-tracking and saturated regimes are both visible.

Forward check: the zeros are only an *output* being characterised — nothing is
fitted to them.

    uv run python scripts/run_number_variance.py                    # 100k zeros, GPU
    uv run python scripts/run_number_variance.py --source first-2M --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _stats(unfolded, lengths, n_offsets, use_gpu):
    if use_gpu:
        try:
            from zeta_spectral_gpu import spacing_gpu

            return (
                spacing_gpu.number_variance_gpu(unfolded, lengths, n_offsets=n_offsets),
                spacing_gpu.dyson_mehta_delta3_gpu(
                    unfolded, lengths, n_offsets=n_offsets
                ),
                "gpu",
            )
        except ImportError:
            pass
    return (
        spacing.number_variance(unfolded, lengths, n_offsets=n_offsets),
        spacing.dyson_mehta_delta3(unfolded, lengths, n_offsets=n_offsets),
        "cpu",
    )


def _benchmark(unfolded, lengths, n_offsets, repeats):
    import cupy as cp

    from zeta_spectral_gpu import spacing_gpu

    t0 = time.perf_counter()
    for _ in range(repeats):
        spacing.number_variance(unfolded, lengths, n_offsets=n_offsets)
        spacing.dyson_mehta_delta3(unfolded, lengths, n_offsets=n_offsets)
    cpu_ms = (time.perf_counter() - t0) / repeats * 1e3

    spacing_gpu.number_variance_gpu(unfolded, lengths, n_offsets=n_offsets)  # warm up
    spacing_gpu.dyson_mehta_delta3_gpu(unfolded, lengths, n_offsets=n_offsets)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        spacing_gpu.number_variance_gpu(unfolded, lengths, n_offsets=n_offsets)
        spacing_gpu.dyson_mehta_delta3_gpu(unfolded, lengths, n_offsets=n_offsets)
    cp.cuda.Device().synchronize()
    gpu_ms = (time.perf_counter() - t0) / repeats * 1e3
    return cpu_ms, gpu_ms


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
    ap.add_argument("--l-min", type=float, default=1.0, help="smallest window length")
    ap.add_argument("--l-max", type=float, default=40.0, help="largest window length")
    ap.add_argument("--n-l", type=int, default=36, help="number of window lengths")
    ap.add_argument(
        "--n-offsets", type=int, default=4000, help="windows averaged per L"
    )
    ap.add_argument(
        "--out", type=Path, default=DATA / "rigidity.png", help="figure path"
    )
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU path")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    n = args.n
    if args.source == "mpmath" and n is None:
        n = 5000

    tau = zeros.load_ordinates(n, source=args.source, cache_dir=DATA)
    unfolded = zeros.unfold(tau)
    lengths = np.geomspace(args.l_min, args.l_max, args.n_l)
    # Berry saturation scale at the representative height: L* ~ ln(T/2pi)/pi.
    saturation_l = float(np.log(np.median(tau) / (2.0 * np.pi)) / np.pi)

    sigma2, delta3, backend = _stats(
        unfolded, lengths, args.n_offsets, use_gpu=not args.no_gpu
    )

    print(f"source={args.source}  N={tau.size:,}  backend={backend}")
    print(
        f"  L in [{args.l_min:g}, {args.l_max:g}]  ({args.n_l} pts)  "
        f"n_offsets={args.n_offsets}"
    )
    print(
        f"  Sigma^2(L_max)={sigma2[-1]:.4f}  (GUE "
        f"{float(spacing.gue_number_variance(args.l_max)):.4f}, "
        f"Poisson {args.l_max:.4f})"
    )
    print(
        f"  Delta_3(L_max)={delta3[-1]:.4f}  (GUE "
        f"{float(spacing.gue_delta3(args.l_max)):.4f}, "
        f"Poisson {args.l_max / 15.0:.4f})"
    )
    print(f"  Berry saturation scale L* ~ ln(T/2pi)/pi = {saturation_l:.2f}")

    if backend == "gpu":
        from zeta_spectral_gpu import spacing_gpu

        m = min(8000, tau.size)
        sub = unfolded[:m]
        sub_lengths = lengths[lengths < (sub[-1] - sub[0])]
        cpu_v = spacing.number_variance(sub, sub_lengths, n_offsets=args.n_offsets)
        cpu_d = spacing.dyson_mehta_delta3(sub, sub_lengths, n_offsets=args.n_offsets)
        gpu_v = spacing_gpu.number_variance_gpu(
            sub, sub_lengths, n_offsets=args.n_offsets
        )
        gpu_d = spacing_gpu.dyson_mehta_delta3_gpu(
            sub, sub_lengths, n_offsets=args.n_offsets
        )
        print(
            f"  GPU vs CPU on first {m}: "
            f"max|dSigma^2|={np.max(np.abs(cpu_v - gpu_v)):.2e}  "
            f"max|dDelta_3|={np.max(np.abs(cpu_d - gpu_d)):.2e}"
        )

    out = plots.rigidity_figure(
        lengths,
        sigma2,
        delta3,
        n_levels=tau.size,
        out_path=args.out,
        saturation_l=saturation_l,
        title=f"Spectral rigidity of the Riemann zeros vs GUE  (N={tau.size:,})",
    )
    print(f"  figure -> {out}")

    if args.benchmark:
        if backend != "gpu":
            print("  (benchmark needs the GPU path; skipped)")
        else:
            cpu_ms, gpu_ms = _benchmark(unfolded, lengths, args.n_offsets, args.repeats)
            print(f"  benchmark over {args.repeats} runs (N={tau.size:,}):")
            print(f"    CPU reference {cpu_ms:9.3f} ms")
            print(f"    GPU sweep     {gpu_ms:9.3f} ms   (x{cpu_ms / gpu_ms:.1f})")
            if cpu_ms / gpu_ms < 1.0:
                print(
                    "    note: the GPU is launch-bound at this size; its win grows\n"
                    "          with N and --n-offsets (e.g. first-2M is multiple x)."
                )


if __name__ == "__main__":
    main()
