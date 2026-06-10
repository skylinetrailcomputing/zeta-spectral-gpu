"""Experiment #84: arithmetic beyond universality — BK/CS lower-order pair
correlation and the spectral form factor of the zeros.

Two forward readouts over a height window of precomputed zeros:

* **Form factor**: the windowed Fourier statistic ``S(u) = sum w(tau) e^{iu tau}``
  of the zero ordinates vs the explicit-formula prediction assembled purely
  from primes (``Lambda(n)/sqrt(n)`` times the window transform at
  ``u = log p^m``) — the primes literally visible as peaks in the zeros'
  Fourier statistics, orders of magnitude above the GUE diagonal ramp.
* **Pair correlation**: the raw-separation pair histogram vs the Conrey-Snaith
  ratios-conjecture prediction (Theorem 4.1 — equal to the Bogomolny-Keating
  Hardy-Littlewood form), whose departure from the bare sine kernel is the
  arithmetic content GUE universality cannot see.

Forward check: predictions consume primes and zeta on the 1-line only; the
zeros are only ever the *output being characterised*.

    uv run python scripts/run_arithmetic_correlations.py                  # 100k zeros
    uv run python scripts/run_arithmetic_correlations.py --source first-2M --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import arithmetic_correlations as ac
from zeta_spectral_gpu import plots, spacing, zeros

DATA = Path(__file__).resolve().parent.parent / "data"


def _zero_fourier(tau, u, t_lo, t_hi, window, use_gpu):
    if use_gpu:
        try:
            from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

            return acg.zero_fourier_gpu(tau, u, t_lo, t_hi, window=window), "gpu"
        except ImportError:
            pass
    return ac.zero_fourier(tau, u, t_lo, t_hi, window=window), "cpu"


def _prime_marks(u_min: float, u_max: float) -> list[tuple[float, str]]:
    """(log p^m, label) marks for every prime power in the frequency range."""
    lam = ac.von_mangoldt(int(np.exp(u_max)) + 1)
    marks = []
    for n in np.nonzero(lam)[0]:
        pos = float(np.log(n))
        if u_min <= pos <= u_max:
            marks.append((pos, str(int(n))))
    return marks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source", default="first-100k", help="'mpmath' or an Odlyzko table"
    )
    ap.add_argument(
        "--t-lo", type=float, default=None, help="window low edge (default 0.4*max)"
    )
    ap.add_argument(
        "--t-hi", type=float, default=None, help="window high edge (default 0.99*max)"
    )
    ap.add_argument("--window", default="hann", choices=("hann", "rect"))
    ap.add_argument("--u-min", type=float, default=0.25)
    ap.add_argument("--u-max", type=float, default=4.0)
    ap.add_argument("--n-u", type=int, default=6000)
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--max-sep", type=float, default=8.0)
    ap.add_argument(
        "--p-max", type=int, default=ac.DEFAULT_P_MAX, help="prime cutoff for A, B"
    )
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--no-gpu", action="store_true", help="force the (slow) CPU path")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    n = 2000 if args.source == "mpmath" else None
    tau = zeros.load_ordinates(n, source=args.source, cache_dir=DATA)
    t_lo = args.t_lo if args.t_lo is not None else 0.4 * float(tau[-1])
    t_hi = args.t_hi if args.t_hi is not None else 0.99 * float(tau[-1])
    sel = tau[(tau >= t_lo) & (tau <= t_hi)]
    d_bar = np.log(0.5 * (t_lo + t_hi) / (2 * np.pi)) / (2 * np.pi)
    print(
        f"source={args.source}  N={tau.size:,}  window=[{t_lo:.0f}, {t_hi:.0f}] "
        f"({sel.size:,} zeros, dbar={d_bar:.3f}, taper={args.window})"
    )

    # --- Form factor ---------------------------------------------------------
    u = np.linspace(args.u_min, args.u_max, args.n_u)
    s, backend = _zero_fourier(tau, u, t_lo, t_hi, args.window, not args.no_gpu)
    s2_emp = np.abs(s) ** 2
    pred = ac.prime_prediction(u, t_lo, t_hi, window=args.window)
    s2_pred = np.abs(pred) ** 2
    ramp = ac.diagonal_ramp(u, t_lo, t_hi, window=args.window)
    print(f"  form factor backend={backend}  grid={u.size} points")

    if backend == "gpu":
        from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

        u_sub = u[:: max(1, u.size // 64)]
        cpu_sub = ac.zero_fourier(tau[:5000], u_sub, t_lo, t_hi, window=args.window)
        gpu_sub = acg.zero_fourier_gpu(
            tau[:5000], u_sub, t_lo, t_hi, window=args.window
        )
        agree = float(np.abs(cpu_sub - gpu_sub).max())
        print(f"  GPU vs CPU on subset: max abs diff {agree:.3e}")

    print("  prime-power peaks (|S|^2 at u = log n):")
    print("      n     predicted     empirical     ratio")
    for pos, label in _prime_marks(args.u_min, args.u_max)[:12]:
        s_pk = ac.zero_fourier(tau, np.array([pos]), t_lo, t_hi, window=args.window)
        p_pk = ac.prime_prediction(np.array([pos]), t_lo, t_hi, window=args.window)
        e2, p2 = float(np.abs(s_pk[0]) ** 2), float(np.abs(p_pk[0]) ** 2)
        print(f"   {label:>4}  {p2:12.5g}  {e2:12.5g}  {e2 / p2:9.4f}")

    out1 = plots.zero_form_factor_figure(
        u,
        s2_emp,
        s2_pred,
        ramp,
        _prime_marks(args.u_min, args.u_max),
        out_path=args.out_dir / "arithmetic_form_factor.png",
        title=(
            f"Form factor of {sel.size:,} zeros in [{t_lo:.0f}, {t_hi:.0f}] "
            f"vs the prime prediction"
        ),
    )
    print(f"  figure -> {out1}")

    # --- Pair correlation deviations ----------------------------------------
    hist = spacing.pair_correlation_histogram(sel, args.bin_width, args.max_sep)
    eps = (np.arange(hist.size) + 0.5) * args.bin_width
    emp = hist / args.bin_width  # ordered pairs per unit separation
    cs = ac.cs_pair_density(eps, t_lo, t_hi, p_max=args.p_max)
    gue = ac.gue_pair_density(eps, t_lo, t_hi)
    from scipy.integrate import simpson

    t_grid = np.linspace(t_lo, t_hi, 513)
    plateau = float(
        simpson((np.log(t_grid / (2 * np.pi)) / (2 * np.pi)) ** 2, x=t_grid)
    )

    mask = eps > 2.0 / (d_bar * np.pi**2)  # past the strong-repulsion knee
    dev_emp, dev_cs = emp[mask] - gue[mask], cs[mask] - gue[mask]
    rms_gue = float(np.sqrt(np.mean((dev_emp / plateau) ** 2)))
    rms_cs = float(np.sqrt(np.mean(((emp[mask] - cs[mask]) / plateau) ** 2)))
    corr = float(np.corrcoef(dev_emp, dev_cs)[0, 1])
    shot = float(
        np.sqrt(np.mean(emp[mask] / args.bin_width)) / plateau
    )  # Poisson noise per bin
    print(f"  R2 over {sel.size:,} zeros, bins of {args.bin_width}:")
    print(f"    rms deviation from GUE-only      {rms_gue:.5f}")
    print(
        f"    rms residual vs Conrey-Snaith    {rms_cs:.5f}  (shot noise ~{shot:.5f})"
    )
    print(f"    corr[(zeros - GUE), (CS - GUE)]  {corr:+.3f}")

    out2 = plots.pair_correlation_deviation_figure(
        eps,
        emp,
        cs,
        gue,
        plateau,
        out_path=args.out_dir / "arithmetic_pair_correlation.png",
        title=(
            f"Pair correlation of {sel.size:,} zeros in [{t_lo:.0f}, {t_hi:.0f}]: "
            f"arithmetic terms beyond GUE"
        ),
    )
    print(f"  figure -> {out2}")

    if args.benchmark:
        if backend != "gpu":
            print("  (benchmark needs the GPU path; skipped)")
        else:
            import cupy as cp

            from zeta_spectral_gpu import arithmetic_correlations_gpu as acg

            t0 = time.perf_counter()
            ac.zero_fourier(tau, u, t_lo, t_hi, window=args.window)
            cpu_ms = (time.perf_counter() - t0) * 1e3
            acg.zero_fourier_gpu(tau, u, t_lo, t_hi, window=args.window)  # warm
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            acg.zero_fourier_gpu(tau, u, t_lo, t_hi, window=args.window)
            cp.cuda.Device().synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1e3
            print(
                f"  benchmark (N={sel.size:,}, {u.size} frequencies): "
                f"CPU {cpu_ms:.1f} ms  GPU {gpu_ms:.1f} ms  (x{cpu_ms / gpu_ms:.0f})"
            )


if __name__ == "__main__":
    main()
