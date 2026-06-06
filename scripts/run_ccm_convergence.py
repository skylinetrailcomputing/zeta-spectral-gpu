"""Issue #65: the CCM spectrum's convergence law toward the zeta zeros.

Forward throughout: the spectrum is the prime-built CCM operator's (``ccm.py``);
the zeros are only the yardstick. Four studies (Sliwinski arXiv:2601.12133):

  artifact   -- Is the "inverse-log" numerics a PRECISION ARTIFACT? Contrast the
                genuine (mpmath) low-zero error -- super-exponential in the cutoff
                (Groskin) -- with the fp64 error, which is dominated by xi-corruption
                and cannot collapse. (Phase-0 headline finding of #65.)
  edge       -- The per-index error profile vs the Heisenberg floor 1/(4 ln lambda)
                (Thm 3.1): low zeros sit far below it (super-exp), the floor is a
                resolution-EDGE phenomenon.
  accelerate -- Exploit the law (F2): Wynn-epsilon / Aitken extrapolation of a
                zero's cutoff-sequence toward x -> infinity, vs the raw best cutoff.
                The honest answer is NEGATIVE: where the convergence is clean (the
                low/resolved zeros) it is super-exponential, so no geometric
                accelerator beats the raw deepest cutoff; the only slow (inverse-log)
                part is the unresolvable resolution edge. Reported because "accel
                does not help, and here is why" is itself the deliverable.

    uv run python scripts/run_ccm_convergence.py                 # all, modest scale
    uv run python scripts/run_ccm_convergence.py --mode artifact
    uv run --extra gpu python scripts/run_ccm_convergence.py     # GPU fp64 assembly

Spectra are cached (full precision) under ``data/`` keyed by ``(N, x, dps)``; the
mpmath eigensolve is the cost, so reruns and replots are instant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import ccm, ccm_convergence as cc, plots

DATA = Path(__file__).resolve().parent.parent / "data"


# ----------------------------------------------------------------------------
# Cached high-precision spectra (the expensive piece)
# ----------------------------------------------------------------------------


def spectrum_mpf(N: int, x: int, dps: int, *, refresh: bool = False) -> list:
    """The operator's positive spectrum at ``(N, x, dps)`` as full-precision mpf.

    Cached as decimal strings so the deep digits survive (needed to accelerate the
    super-exponentially small low-zero errors). Distinct from
    ``run_ccm_gpu.cached_spectrum``, which stores float64 (fine for stats, lossy
    for extrapolation).
    """
    cache = DATA / f"ccm_spec_mpf_N{N}_x{x}_dps{dps}.json"
    if cache.exists() and not refresh:
        with mp.workdps(dps):
            return [mp.mpf(s) for s in json.loads(cache.read_text())]
    with mp.workdps(dps):
        spec = ccm.operator_spectrum(N, mp.sqrt(x), count=N, dps=dps)
        out = [mp.nstr(s, dps) for s in spec]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    with mp.workdps(dps):
        return [mp.mpf(s) for s in out]


def zeros_mpf(count: int, dps: int) -> list:
    cache = DATA / f"zeta_ordinates_{count}_dps{dps}.json"
    if cache.exists():
        with mp.workdps(dps):
            return [mp.mpf(s) for s in json.loads(cache.read_text())]
    with mp.workdps(dps):
        z = ccm.reference_ordinates(count)
        out = [mp.nstr(v, dps) for v in z]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    with mp.workdps(dps):
        return [mp.mpf(s) for s in out]


# ----------------------------------------------------------------------------
# artifact: genuine (super-exp) vs fp64 (corrupted) low-zero error
# ----------------------------------------------------------------------------


def study_artifact(xs: list[int], *, N: int, low: int) -> dict:
    print(f"\n[artifact] genuine vs fp64 error over the first {low} zeros (N={N})")
    print(
        f"{'x':>5} {'lnlam':>6} {'dps':>5} {'genuine_max':>12} {'fp64_max':>10} "
        f"{'corruption':>11}"
    )
    rows = []
    for x in xs:
        dps = cc.suggest_dps(x)
        corr = cc.fp64_spectrum_corruption(N, mp.sqrt(x), low, dps=dps)
        rows.append(
            {
                "x": x,
                "ln_lambda": 0.5 * float(np.log(x)),
                "genuine": corr.max_vs_zeros_mpmath,
                "fp64": corr.max_vs_zeros_fp64,
                "corruption": corr.max_vs_mpmath,
            }
        )
        print(
            f"{x:>5} {rows[-1]['ln_lambda']:>6.2f} {dps:>5} "
            f"{corr.max_vs_zeros_mpmath:>12.2e} {corr.max_vs_zeros_fp64:>10.2e} "
            f"{corr.max_vs_mpmath:>11.2e}"
        )
    return {"rows": rows, "N": N, "low": low}


# ----------------------------------------------------------------------------
# edge: per-index error profile vs the Heisenberg floor
# ----------------------------------------------------------------------------


def study_edge(x: int, *, N: int, dps: int | None) -> dict:
    dps = dps or cc.suggest_dps(x)
    print(
        f"\n[edge] per-index error profile vs Heisenberg floor (N={N}, x={x}, "
        f"dps={dps})"
    )
    spec = spectrum_mpf(N, x, dps)
    zr = zeros_mpf(len(spec), dps)
    with mp.workdps(dps):
        errs = [abs(spec[k] - zr[k]) for k in range(len(spec))]
        bound = float(cc.heisenberg_bound(mp.sqrt(x)))
        mae = float(mp.fsum(errs) / len(errs))
    errs_f = np.array([float(e) for e in errs])
    k_cross = int(np.argmax(errs_f >= bound)) + 1 if (errs_f >= bound).any() else None
    print(
        f"  tracked {len(errs_f)} zeros; first-zero err {errs_f[0]:.2e}, "
        f"edge err {errs_f[-1]:.3f}"
    )
    print(
        f"  Heisenberg floor 1/(4 ln lambda) = {bound:.4f}; "
        f"tracked-set MAE = {mae:.4f} ({'below' if mae < bound else 'above'} floor)"
    )
    if k_cross:
        print(
            f"  per-index error first reaches the floor at index k = {k_cross} "
            f"(the resolution edge)"
        )
    return {
        "x": x,
        "N": N,
        "errors": errs_f.tolist(),
        "bound": bound,
        "mae": mae,
        "k_cross": k_cross,
    }


# ----------------------------------------------------------------------------
# accelerate: extrapolate a moderate zero's cutoff-sequence
# ----------------------------------------------------------------------------


def study_accelerate(xs: list[int], *, N: int, indices: list[int]) -> dict:
    print(
        f"\n[accelerate] Wynn/Aitken extrapolation of nu_k over cutoffs x={xs} (N={N})"
    )
    dps = max(cc.suggest_dps(x) for x in xs)
    specs = {x: spectrum_mpf(N, x, dps) for x in xs}
    zr = zeros_mpf(max(indices), dps)
    print(f"{'k':>4} {'raw_err':>12} {'wynn_err':>12} {'shanks_err':>12} {'gain':>8}")
    rows = []
    with mp.workdps(dps):
        for k in indices:
            seq = [specs[x][k - 1] for x in xs if k - 1 < len(specs[x])]
            if len(seq) < 3:
                continue
            az = cc.accelerate_zero(k, seq, xs, zr[k - 1])
            rows.append(
                {
                    "index": k,
                    "raw": float(az.raw_error),
                    "wynn": float(az.wynn_error),
                    "shanks": float(az.shanks_error),
                    "gain": float(az.gain) if az.gain != mp.inf else float("inf"),
                }
            )
            print(
                f"{k:>4} {float(az.raw_error):>12.2e} {float(az.wynn_error):>12.2e} "
                f"{float(az.shanks_error):>12.2e} {rows[-1]['gain']:>8.2f}"
            )
    helped = [r for r in rows if r["gain"] > 1.5]
    print(
        f"  -> acceleration beat the raw best cutoff (gain>1.5) for "
        f"{len(helped)}/{len(rows)} indices: super-exp convergence leaves nothing "
        f"for a geometric accelerator to exploit."
    )
    return {"rows": rows, "xs": xs, "N": N}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=["artifact", "edge", "accelerate", "all"],
        default="all",
    )
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.mode in ("artifact", "all"):
        a = study_artifact([11, 12, 13, 14, 15], N=args.N, low=12)
        out = plots.ccm_convergence_artifact_figure(
            a, out_path=args.out_dir / "ccm_convergence_artifact.png"
        )
        print(f"  figure -> {out}")
    if args.mode in ("edge", "all"):
        e = study_edge(13, N=max(args.N, 120), dps=None)
        out = plots.ccm_convergence_edge_figure(
            e, out_path=args.out_dir / "ccm_convergence_edge.png"
        )
        print(f"  figure -> {out}")
    if args.mode in ("accelerate", "all"):
        study_accelerate([11, 12, 13, 14], N=args.N, indices=[10, 20, 30, 40])

    print(f"\n[done] {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
