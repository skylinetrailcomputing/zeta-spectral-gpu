"""Issue #52 (+ #71 family): Li's criterion as a forward, computable RH probe.

Forward throughout: the Li coefficients ``lambda_n`` are computed from the Taylor
coefficients of ``log xi(s)`` at ``s = 1`` (Stieltjes constants + polygamma), never
from the ``sum_rho`` over zeros. RH is equivalent to ``lambda_n >= 0`` for all ``n``;
the zeros enter only as the positivity prediction we compare against. A scalar
shadow of the flagship CCM operator's Weil positivity ``lambda_min(c) >= 0``.

    uv run python scripts/run_li_criterion.py            # n <= 40 (single zeta)
    uv run python scripts/run_li_criterion.py --N 80 --dps 130

The ``--family`` mode is the Phase-2 generalisation (#71): the **Generalized** RH for
a Dirichlet ``L``-function ``L(s, chi)`` is equivalent to ``Re lambda_n(chi) >= 0``,
computed forward from ``log Lambda(s, chi)``. A whole family of characters is
embarrassingly parallel -- the GPU assembles the family in fp64 (one block per
character), with mpmath as the small-``n`` reference:

    uv run python scripts/run_li_criterion.py --family quadratic --qmax 24 --N 20
    uv run python scripts/run_li_criterion.py --family prime --qmax 13 --N 16

Precision grows with ``n`` (binomial cancellation + Stieltjes digits); ``--dps``
defaults to ``li_criterion.default_dps`` and the run reports a stability residual.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mpmath as mp

from zeta_spectral_gpu import li_criterion as li, plots

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--N", type=int, default=40, help="highest Li index n")
    ap.add_argument("--dps", type=int, default=None, help="mpmath working precision")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--family",
        choices=["quadratic", "prime", "all"],
        default=None,
        help="run the Phase-2 GRH family sweep (#71) instead of the single-zeta sweep",
    )
    ap.add_argument(
        "--qmax", type=int, default=12, help="family conductor/modulus bound"
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="force the mpmath path; skip the GPU family assembly",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.family:
        run_family(args)
    else:
        run_single(args)


def run_single(args) -> None:
    t0 = time.perf_counter()
    res = li.evaluate(args.N, dps=args.dps)

    print(
        f"\n[li] forward Li coefficients from log xi  (n<={res.n_max}, dps={res.dps})"
    )
    print(f"{'n':>4} {'lambda_n':>20} {'main(n)':>14} {'ratio':>9}")
    with mp.workdps(res.dps):
        for n in range(1, res.n_max + 1):
            lam = res.coefficients[n - 1]
            if n == 1:
                print(f"{n:>4} {mp.nstr(lam, 14):>20} {'-':>14} {'-':>9}")
                continue
            main = li.li_main_term(n)
            print(
                f"{n:>4} {mp.nstr(lam, 14):>20} {mp.nstr(main, 8):>14} "
                f"{float(lam / main):>9.4f}"
            )

    verdict = (
        "CONSISTENT with RH" if res.rh_consistent else "NEGATIVE -> would REFUTE RH"
    )
    print(
        f"\n  min lambda_n = {mp.nstr(res.min_value, 8)} at n={res.min_index}  "
        f"-> {verdict} over n<={res.n_max}"
    )
    print(f"  positivity: {'all positive' if res.all_positive else 'NEGATIVE FOUND'}")
    print(
        f"  stability residual (vs lower-precision recompute) = "
        f"{mp.nstr(res.stability, 3)}"
        + (
            "  [OK]"
            if res.stability < mp.mpf("1e-12")
            else "  [UNDER-RESOLVED: raise --dps]"
        )
    )

    if not args.no_plot:
        out = plots.li_coefficients_figure(
            res, out_path=args.out_dir / "li_coefficients.png"
        )
        print(f"  figure -> {out}")

    print(f"\n[done] {time.perf_counter() - t0:.1f}s")


def run_family(args) -> None:
    from zeta_spectral_gpu import li_criterion_family as fam

    chars = fam.dirichlet_family(args.family, args.qmax)
    if not chars:
        print(
            f"[li-family] no characters in the {args.family} family up to {args.qmax}"
        )
        return
    n_max = args.N
    dps = args.dps or li.default_dps(n_max)
    print(
        f"\n[li-family] forward GRH Li sweep over the {args.family} family "
        f"({len(chars)} characters, n<={n_max}, dps={dps})"
    )

    t0 = time.perf_counter()
    res = fam.evaluate_family(chars, n_max, dps=dps, kind=args.family)  # the reference
    t_cpu = time.perf_counter() - t0

    print(
        f"\n{'character':>12} {'q':>4} {'par':>3} {'real':>5} "
        f"{'min Re lambda_n':>18} {'at n':>5}"
    )
    for m in res.members:
        print(
            f"{m.label:>12} {m.modulus:>4} {m.parity:>3} {str(m.is_real):>5} "
            f"{mp.nstr(m.min_re, 8):>18} {m.min_re_index:>5}"
        )

    w = res.worst_member
    verdict = (
        "CONSISTENT with GRH" if res.all_positive else "NEGATIVE -> would REFUTE GRH"
    )
    print(f"\n  family verdict: {verdict} over {res.n_members} characters, n<={n_max}")
    print(
        f"  tightest GRH margin: min Re lambda = {mp.nstr(w.min_re, 6)} "
        f"at {w.label} (n={w.min_re_index})"
    )
    print(
        f"  max imag residual over real members = {mp.nstr(res.max_imag_residual, 3)}"
        + ("  [OK ~0]" if res.max_imag_residual < mp.mpf("1e-10") else "  [check]")
    )

    # GPU scale readout: the family assembly batched in fp64, one block per character.
    if not args.cpu:
        try:
            from zeta_spectral_gpu import li_criterion_family_gpu as gpu

            tp = time.perf_counter()
            inputs = gpu.prepare_inputs(chars, n_max, dps=dps)
            t_prep = time.perf_counter() - tp
            lam, _, _ = gpu.assemble_gpu(inputs)  # warm up (NVRTC JIT + module load)
            reps = 20
            tg = time.perf_counter()
            for _ in range(reps):
                lam, _, _ = gpu.assemble_gpu(inputs)
            t_gpu = (time.perf_counter() - tg) / reps
            fps = len(chars) / t_gpu if t_gpu > 0 else float("inf")
            ncmp = min(n_max, 8)  # fp64-reliable window for the agreement check
            worst = max(
                abs(complex(res.members[c].coefficients[n]) - lam[c, n])
                for c in range(len(chars))
                for n in range(ncmp)
            )
            print(
                f"\n  GPU assembly: {fps:,.0f} families/sec "
                f"(mpmath prep {t_prep:.2f}s, kernel {t_gpu * 1e3:.2f}ms); "
                f"GPU-vs-CPU max|d lambda| (n<={ncmp}) = {worst:.1e}"
            )
        except ImportError:
            print("\n  GPU assembly: cupy not installed (mpmath reference only)")

    print(f"  mpmath reference time: {t_cpu:.1f}s")

    if not args.no_plot:
        out = plots.li_family_figure(
            res, out_path=args.out_dir / f"li_family_{args.family}.png"
        )
        print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
