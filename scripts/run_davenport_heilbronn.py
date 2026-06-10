"""#85: Davenport-Heilbronn negative control — run the forward machinery on a
functional-equation function that provably violates RH.

f(s) = 1 + kappa/2^s - kappa/3^s - 1/4^s + 1/6^s + ...  (period 5, no Euler
product) satisfies an exact Riemann-type functional equation but has zeros off
the critical line. Three readouts check that the repo's forward pipeline can
tell f from a genuine L-function (a pipeline that cannot is not measuring
arithmetic):

  census  — compute f's off-line zeros (output, then validated against the
            published tables) and the on-line deficit they leave behind.
  growth  — the #43 growth-law discriminator on a GENUINE off-line zero.
  stats   — spacing / ratio statistics of f's on-line zeros vs the component
            L-functions and their superposition.
  locator — the #42 mirror locator with the Dirichlet-inverse weights of 1/f.

    uv run python scripts/run_davenport_heilbronn.py
    uv run python scripts/run_davenport_heilbronn.py --mode stats --t-max 2000
    uv run python scripts/run_davenport_heilbronn.py --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import davenport_heilbronn as dh
from zeta_spectral_gpu import dirac_mirror as dm
from zeta_spectral_gpu import dirichlet as dl
from zeta_spectral_gpu import dirichlet_locator as loc
from zeta_spectral_gpu import spacing

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def run_census(t_max: float) -> np.ndarray:
    print(f"-- census: off-line zeros of f with sigma > 1/2, 2 < t < {t_max:g} --")
    off = dh.off_line_zeros(2.0, t_max)
    for rho in off:
        print(f"  rho = {rho.real:.12f} + {rho.imag:.12f} i")
    on = dh.critical_line_zeros(2.0, t_max)
    predicted = float(dh.dh_smooth_count(t_max) - dh.dh_smooth_count(2.0))
    deficit = predicted - on.size
    print(f"  on-line zeros found: {on.size}  smooth count: {predicted:.1f}")
    print(f"  deficit: {deficit:.1f}  (each off-line pair removes 2 from the line;")
    print(f"   {off.size} representatives found here -> {2 * off.size} accounted)")
    return off


def run_growth(off: np.ndarray, n_max: int, plot: bool) -> None:
    print(f"-- growth: #43 discriminator with 1/f inverse weights, n_max={n_max:g} --")
    c = dh.dirichlet_inverse(n_max)
    rho = complex(off[0])
    on = float(dh.critical_line_zeros(49.0, 52.0)[0])
    energies = (
        (f"off-line E={rho.imag:.4f}", rho.imag, rho.real - 0.5),
        (f"on-line E={on:.4f}", on, None),
        ("generic E=51.3000", 51.3, None),
    )
    truncations = np.unique(np.geomspace(100, n_max, 400).astype(np.int64))
    profiles: dict[str, np.ndarray] = {}
    slopes: dict[str, float] = {}
    for label, E, predicted in energies:
        profile = dh.partial_sum_profile(E, c, truncations)
        slope = dh.loglog_rms_slope(truncations, profile)
        key = label.split(" E=")[0]
        profiles[key], slopes[key] = profile, slope
        suffix = f"  (predicted {predicted:.4f})" if predicted is not None else ""
        print(f"  {label}: slope {slope:+.4f}{suffix}")
    print("  (Moebius weights on a genuine L give a bounded profile, slope ~ 0;")
    print("   the ~+0.1 background is itself the no-Euler-product signature)")
    if plot:
        from zeta_spectral_gpu import plots

        study = {
            "truncations": truncations,
            "profiles": profiles,
            "slopes": slopes,
            "predicted": rho.real - 0.5,
        }
        out = plots.davenport_heilbronn_growth_figure(
            study, out_path=FIGURES / "davenport_heilbronn_growth.png"
        )
        print(f"  figure -> {out}")


def run_stats(t_max: float, plot: bool) -> None:
    print(f"-- stats: on-line zero statistics to t = {t_max:g} --")
    z_f = dh.critical_line_zeros(5.0, t_max)
    z_chi = dh.critical_line_zeros(5.0, t_max, which="chi")
    z_chibar = dh.critical_line_zeros(5.0, t_max, which="chibar")
    union = np.sort(np.concatenate([z_chi, z_chibar]))
    predicted = float(dh.dh_smooth_count(t_max) - dh.dh_smooth_count(5.0))
    deficit = predicted - z_f.size

    rtilde = {
        "f": float(np.mean(spacing.spacing_ratios(z_f))),
        "chi": float(np.mean(spacing.spacing_ratios(z_chi))),
        "union": float(np.mean(spacing.spacing_ratios(union))),
    }
    print(f"  zeros: f {z_f.size}, chi {z_chi.size}, chibar {z_chibar.size}")
    print(f"  f deficit vs smooth count: {deficit:.1f}")
    print(
        f"  <r~>: f {rtilde['f']:.4f}, chi {rtilde['chi']:.4f}, "
        f"superposition {rtilde['union']:.4f} "
        f"(GUE {dh.MEAN_RATIO_GUE}, Poisson {dh.MEAN_RATIO_POISSON:.4f})"
    )
    s_f = np.diff(dh.dh_smooth_count(z_f))
    s_union = np.diff(2.0 * (dh.dh_theta(union) / np.pi))
    print(
        f"  tiny gaps (s < 0.2): f {np.mean(s_f < 0.2):.4f}, "
        f"superposition {np.mean(s_union < 0.2):.4f}"
    )
    print("  (repulsion retained: local statistics alone cannot tell f from a")
    print("   genuine L-function at this height; the deficit and census can)")
    if plot:
        from zeta_spectral_gpu import plots

        study = {
            "spacings_f": s_f / np.mean(s_f),
            "spacings_union": s_union / np.mean(s_union),
            "rtilde": rtilde,
            "deficit": deficit,
            "t_max": t_max,
        }
        out = plots.davenport_heilbronn_stats_figure(
            study, out_path=FIGURES / "davenport_heilbronn_stats.png"
        )
        print(f"  figure -> {out}")


def run_locator(off: np.ndarray, n: int, plot: bool) -> None:
    print(f"-- locator: #42 scan with 1/f inverse weights vs chi*mu, n={n:g} --")
    grid = np.arange(2.0, 120.0, 0.01)
    threshold = loc.peak_threshold(n)
    c = dh.dirichlet_inverse(n)
    abs_f, backend = loc.scan_locator(
        dl.dirichlet_character(5, 1), n, grid, weights=c[1:]
    )
    abs_f = np.abs(abs_f)
    chi = dl.dirichlet_character(5, 1)
    abs_chi = np.abs(
        dm.mobius_partial_sum(grid, n, weights=dl.lfunction_weights(chi, n))
    )
    z_f = dh.critical_line_zeros(2.0, 120.0)
    z_chi = dh.critical_line_zeros(2.0, 120.0, which="chi")

    print(f"  backend: {backend}")
    for label, abs_m, true in (("1/f", abs_f, z_f), ("chi*mu", abs_chi, z_chi)):
        peaks = grid[loc.local_maxima(abs_m, threshold)]
        matched = sum(m.matched for m in loc.match_peaks(peaks, true, tol=0.3))
        print(
            f"  {label}: median|M| {np.median(abs_m):.2f}  max {abs_m.max():.1f}  "
            f"peaks {peaks.size}  matched {matched}/{true.size}  "
            f"false {peaks.size - matched}"
        )
    print("  off-line ordinates (expect spurious mounds on the 1/f trace):")
    print(f"   {', '.join(f'{rho.imag:.3f}' for rho in off if rho.imag < 120.0)}")
    if plot:
        from zeta_spectral_gpu import plots

        study = {
            "grid": grid,
            "abs_f": abs_f,
            "abs_chi": abs_chi,
            "threshold": threshold,
            "true_f": z_f,
            "true_chi": z_chi,
            "off_line": [rho.imag for rho in off if rho.imag < 120.0],
        }
        out = plots.davenport_heilbronn_locator_figure(
            study, out_path=FIGURES / "davenport_heilbronn_locator.png"
        )
        print(f"  figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=("census", "growth", "stats", "locator", "all"),
        default="all",
    )
    ap.add_argument("--t-max", type=float, default=1000.0, help="stats zero harvest")
    ap.add_argument("--census-t-max", type=float, default=200.0, help="census box")
    ap.add_argument("--n-max", type=int, default=1_000_000, help="growth truncation")
    ap.add_argument("--locator-n", type=int, default=100_000, help="locator truncation")
    ap.add_argument("--plot", action="store_true", help="save figures under figures/")
    args = ap.parse_args()

    kappa = dh.dh_kappa()
    print(f"Davenport-Heilbronn control: kappa = {kappa:.12f}\n")
    off = None
    if args.mode in ("census", "growth", "locator", "all"):
        off = run_census(args.census_t_max)
    if args.mode in ("growth", "all"):
        run_growth(off, args.n_max, args.plot)
    if args.mode in ("stats", "all"):
        run_stats(args.t_max, args.plot)
    if args.mode in ("locator", "all"):
        run_locator(off, args.locator_n, args.plot)


if __name__ == "__main__":
    main()
