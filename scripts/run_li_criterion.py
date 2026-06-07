"""Issue #52: Li's criterion as a forward, computable RH probe.

Forward throughout: the Li coefficients ``lambda_n`` are computed from the Taylor
coefficients of ``log xi(s)`` at ``s = 1`` (Stieltjes constants + polygamma), never
from the ``sum_rho`` over zeros. RH is equivalent to ``lambda_n >= 0`` for all ``n``;
the zeros enter only as the positivity prediction we compare against. A scalar
shadow of the flagship CCM operator's Weil positivity ``lambda_min(c) >= 0``.

    uv run python scripts/run_li_criterion.py            # n <= 40
    uv run python scripts/run_li_criterion.py --N 80 --dps 130

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
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
