"""Flagship #9: GPU assembly + the lambda-sweep convergence / conditioning study.

Two studies, both forward (the operator is built from the primes; the zeros enter
only to *characterise* the output):

1. ``wall`` — the **precision wall**. Sweep the prime cutoff ``x = lambda^2`` and
   compare the minimal eigenvalue ``epsilon_N`` (and the condition number) computed
   exactly in ``mpmath`` against the fp64 ``eigh`` (cuSOLVER on the GPU, or numpy on
   the CPU). ``epsilon_N`` plummets geometrically while fp64 flattens at its
   ``~1e-16`` floor: the empirical shadow of the limit-control problem, and the
   reason the extended-precision eigensolve must stay on the CPU (issue #9 /
   CLAUDE.md). The fp64 matrix itself is assembled on the GPU
   (``ccm_gpu.assemble_weil_matrix_gpu``), reproducing the mpmath fill to ~1e-12.

2. ``universality`` — the **cross-lambda forward prediction** (the #15/#20
   cross-links). For each ``x`` take the operator's computed spectrum (mpmath),
   unfold it by its own counting function, and run the warm-up universality
   diagnostics (#5 spacing, #6 pair correlation, #15 Sigma^2 / Delta_3, on the GPU),
   plus the **unfolding-free spacing ratio ``r~``** (Atas 2013; #18/#35) over the
   full spectrum and the low (zero-tracking) window. The prediction: as ``x`` grows
   (more primes) the operator's local rigidity relaxes toward GUE and the
   GUE-tracking range extends. ``r~`` sidesteps the finite-``N`` unfolding artifact
   that washes out ``Sigma^2``; ``--pushn`` adds the control showing why enlarging
   ``N`` (the pole-locked tail) is the wrong lever. See
   ``knowledge/ccm-universality.md``.

    uv run --extra gpu python scripts/run_ccm_gpu.py                 # both studies
    uv run --extra gpu python scripts/run_ccm_gpu.py --mode wall
    uv run python scripts/run_ccm_gpu.py --mode universality \
        --x 6 9 12 14 18 24 --N 160 --pushn      # the #18 r~ cross-cutoff read

Spectra and the conditioning sweep are cached under ``data/`` keyed by
``(N, x, dps)`` so reruns (and replotting) are instant. The mpmath eigensolve is
the cost; the GPU does the fill, the conditioning probe, and the statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import ccm, ccm_gpu, debruijn_newman, plots, spacing

DATA = Path(__file__).resolve().parent.parent / "data"


def _gpu_ok() -> bool:
    """True if CuPy + cuSOLVER are usable; the studies fall back to numpy fp64."""
    try:
        cp = ccm_gpu._cupy()
        cp.linalg.eigh(cp.eye(2, dtype=cp.float64))
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


# ----------------------------------------------------------------------------
# Study 1: the precision wall (convergence + conditioning)
# ----------------------------------------------------------------------------


def conditioning_sweep(
    x_values: list[int], *, N: int, dps: int, use_gpu: bool, refresh: bool = False
) -> dict:
    """ε_N and condition number vs cutoff: mpmath (exact) vs fp64 eigh.

    Cached to ``data/ccm_wall_N{N}_dps{dps}.json``. ``eps_mpmath`` is the exact
    minimal eigenvalue; ``eps_fp64`` / ``cond_fp64`` come from the fp64 eigensolve
    (GPU cuSOLVER if available, else numpy). The exact condition number uses the
    fp64 largest eigenvalue (well-conditioned) over the exact ``eps_mpmath``.
    """
    cache = DATA / f"ccm_wall_N{N}_dps{dps}.json"
    store: dict[str, dict] = {}
    if cache.exists() and not refresh:
        store = json.loads(cache.read_text())

    rows = []
    for x in x_values:
        key = str(x)
        if key not in store:
            with mp.workdps(dps):
                A = ccm.assemble_weil_matrix(N, mp.sqrt(x))
                eps_mp = ccm.smallest_even_eigenvector(A, N).eigenvalue
            if use_gpu:
                Afp = ccm_gpu.assemble_weil_matrix_gpu(N, float(np.sqrt(x)))
                cond = ccm_gpu.conditioning_fp64(Afp)
                eps_fp64, eig_max = cond.eps_n, cond.eig_max
            else:
                Afp = ccm_gpu.assemble_weil_matrix_fp64(N, float(np.sqrt(x)))
                w = np.abs(np.linalg.eigvalsh(Afp))
                eps_fp64, eig_max = float(w.min()), float(w.max())
            store[key] = {
                "eps_mpmath": mp.nstr(eps_mp, 17),
                "eps_fp64": eps_fp64,
                "eig_max": eig_max,
            }
        d = store[key]
        eps_mp = mp.mpf(d["eps_mpmath"])
        rows.append(
            {
                "x": x,
                "eps_mpmath": float(eps_mp),
                "eps_fp64": d["eps_fp64"],
                "cond_mpmath": d["eig_max"] / float(eps_mp),
                "cond_fp64": d["eig_max"] / d["eps_fp64"],
            }
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(store, indent=2))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ----------------------------------------------------------------------------
# Study 2: the universality / forward-prediction study
# ----------------------------------------------------------------------------


def cached_spectrum(N: int, x: int, dps: int, *, refresh: bool = False) -> np.ndarray:
    """The operator's positive spectrum at ``(N, x, dps)``, cached as float64 CSV."""
    cache = DATA / f"ccm_spectrum_N{N}_x{x}_dps{dps}.csv"
    if cache.exists() and not refresh:
        return np.atleast_1d(np.loadtxt(cache))
    spec = ccm.operator_spectrum(N, mp.sqrt(x), count=N, dps=dps)
    out = np.array([float(s) for s in spec], dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(cache, out)
    return out


def universality_study(
    x_values: list[int],
    *,
    N: int,
    dps: int,
    use_gpu: bool,
    unfold_degree: int = 6,
    low_count: int = 40,
    refresh: bool = False,
) -> dict:
    """Spectrum -> rigidity statistics, per cutoff ``x``.

    Two readouts. ``Sigma^2``/``Delta_3`` need the spectrum unfolded by its own
    smooth count (the finite-``N`` weak link, #9/#20). The **spacing ratio**
    ``<r~>`` (Atas 2013; :func:`spacing.spacing_ratios`) is taken on the *raw*
    levels — unfolding-free — over the full spectrum and over the low ``low_count``
    (zero-tracking) window, so it sidesteps that artifact (#18/#35).
    """
    if use_gpu:
        from zeta_spectral_gpu import spacing_gpu

        n_var = spacing_gpu.number_variance_gpu
        d3 = spacing_gpu.dyson_mehta_delta3_gpu
    else:
        n_var = spacing.number_variance
        d3 = spacing.dyson_mehta_delta3

    lengths = np.geomspace(0.5, 30.0, 24)
    sigma2_by_x: dict[float, np.ndarray] = {}
    delta3_by_x: dict[float, np.ndarray] = {}
    spacings_by_x: dict[int, np.ndarray] = {}
    rtilde_full: dict[int, float] = {}
    rtilde_low: dict[int, float] = {}
    for x in x_values:
        t0 = time.perf_counter()
        spec = cached_spectrum(N, x, dps, refresh=refresh)
        spec_sorted = np.sort(spec)
        unfolded = debruijn_newman.empirical_unfold(spec, degree=unfold_degree)
        sigma2_by_x[x] = n_var(unfolded, lengths)
        delta3_by_x[x] = d3(unfolded, lengths)
        spacings_by_x[x] = spacing.nearest_neighbour_spacings(np.sort(unfolded))
        rtilde_full[x] = float(np.nanmean(spacing.spacing_ratios(spec_sorted)))
        nlo = min(spec_sorted.size, low_count)
        rtilde_low[x] = float(np.nanmean(spacing.spacing_ratios(spec_sorted[:nlo])))
        print(
            f"  x={x:>2}: {spec.size} levels, span "
            f"{unfolded.max() - unfolded.min():.1f}   "
            f"<r~> full={rtilde_full[x]:.4f} low{nlo}={rtilde_low[x]:.4f}"
            f"  [{time.perf_counter() - t0:.0f}s]"
        )
    return {
        "lengths": lengths,
        "sigma2_by_x": sigma2_by_x,
        "delta3_by_x": delta3_by_x,
        "spacings_by_x": spacings_by_x,
        "rtilde_full": rtilde_full,
        "rtilde_low": rtilde_low,
        "low_count": low_count,
    }


def pushn_rtilde_study(
    x: int, n_values: list[int], *, dps: int, refresh: bool = False
) -> dict:
    """``<r~>`` of the full CCM spectrum vs the truncation ``N`` at fixed cutoff ``x``.

    The negative control for the universality read (#18): enlarging ``N`` adds
    high-energy levels where the zero density outruns the pole spacing ``2 pi / L``,
    so the secular roots become pole-locked (picket-like) and ``<r~>`` climbs *away*
    from GUE. Confirms the lever is the prime cutoff ``x``, not the matrix size.
    """
    print(f"\n[push-N] <r~> of the full spectrum vs N at fixed x={x}")
    rtilde_by_N: dict[int, float] = {}
    for n in n_values:
        spec = np.sort(cached_spectrum(n, x, dps, refresh=refresh))
        rtilde_by_N[n] = float(np.nanmean(spacing.spacing_ratios(spec)))
        print(f"  N={n:>3}: {spec.size} levels  <r~>={rtilde_by_N[n]:.4f}")
    return {"x": x, "rtilde_by_N": rtilde_by_N}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["wall", "universality", "all"], default="all")
    ap.add_argument("--N", type=int, default=120, help="truncation; dim = 2N+1")
    ap.add_argument(
        "--x",
        type=int,
        nargs="+",
        default=None,
        help="prime cutoffs x = lambda^2 (defaults per study)",
    )
    ap.add_argument("--dps", type=int, default=210, help="mpmath precision")
    ap.add_argument("--wall-N", type=int, default=60, help="N for the wall sweep")
    ap.add_argument("--wall-dps", type=int, default=150, help="dps for the wall sweep")
    ap.add_argument("--refresh", action="store_true", help="ignore caches")
    ap.add_argument(
        "--low-count",
        type=int,
        default=40,
        help="levels in the low (zero-tracking) r~ window",
    )
    ap.add_argument(
        "--pushn",
        action="store_true",
        help="add the push-N r~ control study (slow; large-N spectra)",
    )
    ap.add_argument("--pushn-x", type=int, default=14, help="fixed cutoff for --pushn")
    ap.add_argument(
        "--pushn-N",
        type=int,
        nargs="+",
        default=[60, 120, 180, 240],
        help="truncations N for --pushn",
    )
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    use_gpu = _gpu_ok()
    print(f"GPU (cuSOLVER) available: {use_gpu}")

    if args.mode in ("wall", "all"):
        xs = args.x or list(range(4, 15))
        print(f"\n[wall] eps_N / conditioning sweep  (N={args.wall_N}, x={xs})")
        sweep = conditioning_sweep(
            xs, N=args.wall_N, dps=args.wall_dps, use_gpu=use_gpu, refresh=args.refresh
        )
        for i, x in enumerate(sweep["x"]):
            print(
                f"  x={int(x):>2}: eps_mp={sweep['eps_mpmath'][i]:.2e}  "
                f"eps_fp64={sweep['eps_fp64'][i]:.2e}  "
                f"cond_fp64={sweep['cond_fp64'][i]:.2e}"
            )
        out = plots.ccm_precision_wall_figure(
            sweep["x"],
            sweep["eps_mpmath"],
            sweep["eps_fp64"],
            N=args.wall_N,
            cond_mpmath=sweep["cond_mpmath"],
            cond_fp64=sweep["cond_fp64"],
            out_path=args.out_dir / "ccm_precision_wall.png",
        )
        print(f"  figure -> {out}")

    if args.mode in ("universality", "all"):
        xs = args.x or [6, 9, 12, 14]
        print(f"\n[universality] spectrum -> unfold -> rigidity  (N={args.N}, x={xs})")
        study = universality_study(
            xs,
            N=args.N,
            dps=args.dps,
            use_gpu=use_gpu,
            low_count=args.low_count,
            refresh=args.refresh,
        )
        out = plots.ccm_rigidity_vs_lambda_figure(
            study["lengths"],
            study["sigma2_by_x"],
            study["delta3_by_x"],
            out_path=args.out_dir / "ccm_rigidity_vs_lambda.png",
        )
        print(f"  rigidity figure -> {out}")
        # A representative spacing histogram at the largest cutoff.
        x_top = max(xs)
        sp = plots.spacing_histogram_figure(
            study["spacings_by_x"][x_top],
            out_path=args.out_dir / f"ccm_spacing_x{x_top}.png",
            label=f"CCM spectrum ($x={x_top}$)",
            title=f"CCM operator spacing vs GUE ($x={x_top}$, $N={args.N}$)",
        )
        print(f"  spacing figure  -> {sp}")
        # The unfolding-free spacing-ratio readout (#18/#35): r~ vs cutoff, plus
        # the optional push-N control (r~ vs N at fixed x -> pole-locked tail).
        pushn = (
            pushn_rtilde_study(
                args.pushn_x, args.pushn_N, dps=args.dps, refresh=args.refresh
            )
            if args.pushn
            else None
        )
        rt = plots.ccm_rtilde_vs_cutoff_figure(
            xs,
            study["rtilde_full"],
            study["rtilde_low"],
            low_count=study["low_count"],
            pushn=pushn,
            out_path=args.out_dir / "ccm_rtilde_vs_cutoff.png",
            title=f"CCM operator: spacing-ratio rigidity ($N={args.N}$)",
        )
        print(f"  r~ figure       -> {rt}")


if __name__ == "__main__":
    main()
