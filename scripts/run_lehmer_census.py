"""Stretch #86: small-gap / Lehmer-pair census at height -- the first science
consumer of the #55 GPU Riemann-Siegel evaluator.

Writes the precision budget down first (house rule), then scans height windows
for zeros (GPU kernel if available), reads the small-gap tail against the GUE
repulsion law, runs the Csordas-Smith-Varga Lehmer-pair census with its
per-pair lower bounds on the De Bruijn-Newman constant Lambda, and validates
end-to-end on the classical Lehmer pair near t ~ 7005 (validation only -- no
zero ever enters the pipeline as input).

    uv run python scripts/run_lehmer_census.py
    uv run python scripts/run_lehmer_census.py --quick          # smoke run
    uv run python scripts/run_lehmer_census.py --cosv           # +mpmath demo
        # resolves the COSV 1993 pair at t ~ 3.9e8, which sits BELOW the fp64
        # floor -- the budget's live demonstration (minutes of mpmath).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from zeta_spectral_gpu import lehmer_census as lc
from zeta_spectral_gpu import riemann_siegel as rs
from zeta_spectral_gpu import spacing

# The zero of zeta'(s) between the COSV 1993 Lehmer pair (Stopple 2017, sec. 5);
# used only to aim the validation window -- the pair itself is re-resolved here.
COSV_CENTER = 3.888588860023394e8
COSV_PUBLISHED_LAMBDA = -5.895e-9


def _evaluator(no_gpu: bool):
    if not no_gpu:
        try:
            from zeta_spectral_gpu import riemann_siegel_gpu as gpu

            gpu.hardy_z_gpu(np.array([100.0, 200.0]))  # warm up / probe
            return gpu.hardy_z_gpu, True
        except Exception as exc:  # noqa: BLE001 - CPU-only box: fall back
            print(f"(GPU unavailable: {exc}; using CPU reference)")
    return rs.hardy_z, False


def budget_table(measure: bool) -> list[dict]:
    print("precision budget (fp64 Z error -> resolvable normalized gap floor)")
    print("   height      model err   measured     s_min(A=1)   s_min(A=2.5)")
    rows = []
    import mpmath as mp

    for t in (1.0e5, 1.0e6, 1.0e7, 1.0e8):
        model = float(lc.phase_error_model(t))
        measured = float("nan")
        if measure and t <= 1.0e7:
            ts = t + np.array([0.0, 0.37, 0.81, 1.24, 1.62])
            measured = max(abs(float(rs.hardy_z(x)) - float(mp.siegelz(x))) for x in ts)
        s1 = float(lc.gap_resolution_floor(t))
        s25 = float(lc.gap_resolution_floor(t, z_scale=2.5))
        rows.append({"t": t, "model": model, "measured": measured, "floor": s25})
        print(f"   {t:8.0e}   {model:9.2e}   {measured:9.2e}   {s1:9.2e}    {s25:9.2e}")
    print(
        "   (A = local RMS |Z|; the GUE cube law makes gaps below these floors\n"
        "    ~one-in-1e6 events, so fp64 binds only past t ~ 1e8 -- or for\n"
        "    record-quality pairs like COSV's, see --cosv)"
    )
    return rows


def small_gap_readout(s: np.ndarray, floor: float) -> None:
    n = s.size
    print(f"   small-gap tail vs GUE (n_gaps={n:,}, fp64 floor s_min={floor:.1e}):")
    print("      s<        found    GUE (pi^2/9)s^3 n    Wigner-surmise n")
    grid = np.linspace(0.0, 1.0, 4001)
    wigner_cdf = np.cumsum(spacing.gue_wigner_surmise(grid)) * (grid[1] - grid[0])
    for cut in (0.05, 0.1, 0.2, 0.3, 0.5):
        found = int(np.sum(s < cut))
        gue = float(lc.gue_small_gap_cdf(cut)) * n
        wig = float(np.interp(cut, grid, wigner_cdf)) * n
        print(f"      {cut:4.2f}   {found:8d}   {gue:12.1f}{'':9}{wig:12.1f}")


def census_readout(window: lc.CensusWindow, polish: int) -> list[lc.PairRow]:
    rows = lc.lehmer_census(window)
    n_pairs = window.zeros.size - 1
    lehmer = [r for r in rows if r.lam is not None]
    print(
        f"   census: {len(lehmer)} Lehmer pairs / {n_pairs:,} consecutive pairs "
        f"({100.0 * len(lehmer) / max(n_pairs, 1):.2f}%)"
    )
    print("      gamma-              s       Delta^2 g   lambda<=Lambda   flags")
    for r in rows[:8]:
        lam = f"{r.lam: .3e}" if r.lam is not None else "    --    "
        flag = "BELOW-FLOOR" if r.below_floor else ""
        print(
            f"      {r.gamma_minus:15.6f}  {r.s:7.4f}   {r.delta2g:9.4f}   {lam}   {flag}"
        )
    if lehmer:
        best = lehmer[0]
        print(f"   best forward Lambda bound from this window: {best.lam:.3e}")
    for r in rows[: max(polish, 0)]:
        if r.lam is None:
            continue
        gm, gp = lc.polish_pair(r.gamma_minus, r.gamma_plus, dps=30)
        moved = max(abs(gm - r.gamma_minus), abs(gp - r.gamma_plus))
        g = r.delta2g / (r.gamma_plus - r.gamma_minus) ** 2
        lam_polished = lc.csv_lambda(gp - gm, g)
        print(
            f"   mpmath polish ({r.gamma_minus:.3f}): zeros moved <= {moved:.1e}, "
            f"lambda {r.lam:.6e} -> {lam_polished:.6e}"
        )
    return rows


def classical_pair_validation(evaluator) -> None:
    print("\nvalidation: the classical Lehmer pair (gamma_6709/6710, t ~ 7005)")
    w = lc.scan_zeros(6900.0, 7110.0, evaluator=evaluator)
    rows = lc.lehmer_census(w)
    best = rows[0]
    print(
        f"   found ({best.gamma_minus:.6f}, {best.gamma_plus:.6f})  "
        f"s={best.s:.4f}  Delta^2 g={best.delta2g:.4f}  lambda={best.lam:.3e}"
    )
    print(
        "   (literature: 7005.062866 / 7005.100565 -- after-the-fact check only;\n"
        "    Lehmer-pair quality far under the 4/5 bar, as CSV 1994 first noted)"
    )


def cosv_demo(evaluator) -> None:
    print("\nCOSV 1993 pair at t ~ 3.8886e8 -- the budget's live demonstration")
    t0 = COSV_CENTER
    w = lc.scan_zeros(t0 - 2.0, t0 + 2.0, evaluator=evaluator)
    floor = lc.gap_resolution_floor(t0, z_scale=w.rms_z)
    s_pair = 1.0857e-4 * float(lc.zero_density(t0))  # Stopple sec.-6 data
    print(
        f"   fp64 floor here s_min={floor:.1e}; the pair sits at s~{s_pair:.1e} "
        f"-- {'BELOW' if s_pair < floor else 'above'} the floor"
    )
    print(
        f"   fp64 scan: {w.zeros.size} zeros (expected ~{w.expected_count:.1f}), "
        f"{len(w.near_misses)} near-miss dip(s)"
    )
    print("   resolving with mpmath (this is minutes of arbitrary precision)...")
    t_start = time.perf_counter()
    resolved = lc.resolve_near_miss(t0, span=0.02, n_grid=201, dps=30)
    dt = time.perf_counter() - t_start
    if resolved is None:
        print(f"   no pair resolved at this grid ({dt:.0f}s) -- widen the span?")
        return
    gm, gp = resolved
    delta = gp - gm
    # g from the smooth density alone (lambda is g-insensitive at this quality)
    g = lc.csv_g(np.array([gm, gp]), 0, t_lo=gm - 0.5, t_hi=gp + 0.5)
    lam = lc.csv_lambda(delta, g)
    print(
        f"   resolved pair: ({gm:.7f}, {gp:.7f})  Delta={delta:.4e}  ({dt:.0f}s)\n"
        f"   forward lambda = {lam:.3e}  vs published COSV bound "
        f"{COSV_PUBLISHED_LAMBDA:.3e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--windows",
        default="1e5:6e4,1e6:6e4,1e7:2e4",
        help="comma list of start:span height windows",
    )
    ap.add_argument("--step-fraction", type=float, default=0.05)
    ap.add_argument("--polish", type=int, default=2, help="mpmath-polish top N pairs")
    ap.add_argument("--quick", action="store_true", help="spans / 20 (smoke run)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-cpu-check", action="store_true")
    ap.add_argument("--no-measure", action="store_true", help="skip mpmath err probe")
    ap.add_argument("--cosv", action="store_true", help="resolve the COSV 1993 pair")
    ap.add_argument("--figure", default=None, help="write the census figure here")
    args = ap.parse_args()

    evaluator, have_gpu = _evaluator(args.no_gpu)
    print(f"Lehmer / small-gap census  (evaluator: {'GPU' if have_gpu else 'CPU'})\n")
    budget_table(measure=not args.no_measure)
    classical_pair_validation(evaluator)

    study_windows = []
    for spec in args.windows.split(","):
        start_s, span_s = spec.split(":")
        t_lo = float(start_s)
        span = float(span_s) / (20.0 if args.quick else 1.0)
        t_hi = t_lo + span
        print(f"\nwindow [{t_lo:g}, {t_hi:g}]")
        t0 = time.perf_counter()
        w = lc.scan_zeros(
            t_lo, t_hi, evaluator=evaluator, step_fraction=args.step_fraction
        )
        dt = time.perf_counter() - t0
        deficit = w.expected_count - w.zeros.size
        print(
            f"   {w.zeros.size:,} zeros in {dt:.1f}s  "
            f"(expected ~{w.expected_count:.1f}, deficit {deficit:+.1f}; "
            f"rescued {w.rescued.size}, near-misses {len(w.near_misses)})"
        )
        if abs(deficit) > 3:
            print("   WARNING: count deficit beyond S(t) fluctuation -- missed zeros?")
        if have_gpu and not args.no_cpu_check:
            sub = np.linspace(t_lo, min(t_lo + 50.0, t_hi), 2000)
            agree = float(np.max(np.abs(np.asarray(evaluator(sub)) - rs.hardy_z(sub))))
            print(f"   GPU-vs-CPU on a sub-grid: max|dZ| = {agree:.2e}")

        s = lc.normalized_gaps(w.zeros)
        floor = float(lc.gap_resolution_floor(0.5 * (t_lo + t_hi), z_scale=w.rms_z))
        small_gap_readout(s, floor)
        rows = census_readout(w, args.polish)
        study_windows.append(
            {
                "label": f"t~{t_lo:.0e}",
                "t_lo": t_lo,
                "t_hi": t_hi,
                "s": s,
                "floor": floor,
                "rows": rows,
            }
        )

    if args.cosv:
        cosv_demo(evaluator)

    if args.figure:
        from zeta_spectral_gpu import plots

        out = plots.lehmer_census_figure(
            {"windows": study_windows}, out_path=args.figure
        )
        print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
