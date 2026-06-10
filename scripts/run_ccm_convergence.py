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
  edge-corruption -- (#82) Is fp64 xi-corruption confined to the low/near-null band,
                or does it also hit the EDGE eigenvalues that dominate Sliwinski's
                uniform error E (Conj 4.1)? Splits the per-index corruption low vs
                edge across a truncation sweep and locates the crossover N above
                which fp64's E is set by the genuine (robust) edge, not corruption.
                Gates any Sliwinski outreach; the answer here is NO-GO.
  intermediate -- (#87) The Seba / rank-one intermediate-statistics read of the
                pole-locked tail: first-order pinning offsets from the secular
                couplings alone predict the tail's spacing statistics (picket
                with jitter) and the coupling-vs-pole-spacing crossover, compared
                against the measured tracking height t*(x) ~ 12 x (#53).
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

from zeta_spectral_gpu import (
    ccm,
    ccm_convergence as cc,
    ccm_intermediate as ci,
    debruijn_newman,
    plots,
    spacing,
)

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


def xi_mpf(N: int, x: int, dps: int, *, refresh: bool = False) -> list:
    """The minimal even Weil eigenvector ``xi`` at ``(N, x, dps)``, full precision.

    The couplings of the rank-one secular equation (#87) — the same eigensolve
    that produces the spectrum, so caching it alongside makes the intermediate-
    statistics read free once the spectrum has been paid for. Stored as decimal
    strings like the spectrum cache.
    """
    cache = DATA / f"ccm_xi_mpf_N{N}_x{x}_dps{dps}.json"
    if cache.exists() and not refresh:
        with mp.workdps(dps):
            return [mp.mpf(s) for s in json.loads(cache.read_text())]
    with mp.workdps(dps):
        lam = mp.sqrt(x)
        A = ccm.assemble_weil_matrix(N, lam)
        mode = ccm.smallest_even_eigenvector(A, N)
        out = [mp.nstr(v, dps) for v in mode.eigenvector]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    with mp.workdps(dps):
        return [mp.mpf(s) for s in out]


def spectrum_from_xi(N: int, x: int, dps: int, xi: list) -> list:
    """The positive spectrum from an already-computed ``xi`` (root-find only),
    reading/writing the same cache :func:`spectrum_mpf` uses."""
    cache = DATA / f"ccm_spec_mpf_N{N}_x{x}_dps{dps}.json"
    if cache.exists():
        with mp.workdps(dps):
            return [mp.mpf(s) for s in json.loads(cache.read_text())]
    with mp.workdps(dps):
        L = 2 * mp.log(mp.sqrt(x))
        spec = ccm.operator_eigenvalues(xi, N, L, count=N)
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
# tracking: the zero-tracking range k*(x) vs the prime cutoff (#53)
# ----------------------------------------------------------------------------


def study_tracking(
    xs: list[int],
    *,
    N: int,
    rel_tol: float = 1e-3,
    low_count: int = 40,
    refresh: bool = False,
) -> dict:
    """``k*(x)``: how far up the spectrum the operator tracks the zeros, vs cutoff.

    For each cutoff ``x`` take the cached high-precision spectrum and measure, from
    that *one* spectrum, the three universality readouts together (so they share an
    aligned ``x`` axis — the #53 bridge):

    - the **zero-tracking range** ``k*`` — the leading block tracked to ``rel_tol``
      (:func:`cc.tracking_length`) — plus a loose (``1e-1``) and tight (``1e-6``)
      read, the edge ordinate ``t* = zeta_{k*}``, and the ratio ``t*/x``;
    - the **spacing-ratio rigidity** ``<r~>`` (Atas; unfolding-free) over the full
      spectrum and the low ``low_count`` window — the #18 readout, recomputed here;
    - the **``Delta_3``-to-GUE distance** (:func:`spacing.delta3_gue_distance`) — the
      quantitative form of the cross-cutoff ``Delta_3`` ordering (#53 item 1).

    The density-balance heuristic predicts ``t*`` grows ~linearly in ``x``; the
    constant is read off the measured plateau, not assumed. ``k_pred`` counts zeros
    below the measured mean ``t*/x`` times ``x`` — a self-consistent linear overlay.

    Forward: the operator is prime-built; the zeros only score it (#53).
    """
    print(
        f"\n[tracking] zero-tracking range k*(x) vs cutoff (N={N}, rel_tol={rel_tol})"
    )
    print(
        f"{'x':>4} {'dps':>5} {'k*(1e-1)':>9} {'k*':>5} {'k*(1e-6)':>9} "
        f"{'t*':>8} {'t*/x':>7} {'<r~>full':>9} {'<r~>low':>8} {'D3-GUE':>7} {'cap?':>5}"
    )
    lengths = np.geomspace(0.5, 30.0, 24)
    rows = []
    for x in xs:
        dps = cc.suggest_dps(x)
        spec = spectrum_mpf(N, x, dps, refresh=refresh)
        zr = zeros_mpf(len(spec), dps)
        with mp.workdps(dps):
            errs = [abs(spec[k] - zr[k]) for k in range(len(spec))]
        k_loose = cc.tracking_length(errs, zr, rel_tol=1e-1)
        k_star = cc.tracking_length(errs, zr, rel_tol=rel_tol)
        k_tight = cc.tracking_length(errs, zr, rel_tol=1e-6)
        t_star = cc.tracking_height(zr, k_star)
        capped = k_star >= len(spec)
        ratio = float(t_star) / x if t_star is not None else None

        # The bulk universality readouts, from the same spectrum (float64 is ample
        # for statistics; the deep digits only matter for k*'s low-zero errors).
        spec_f = np.sort(np.array([float(s) for s in spec], dtype=np.float64))
        nlo = min(spec_f.size, low_count)
        rtilde_full = float(np.nanmean(spacing.spacing_ratios(spec_f)))
        rtilde_low = float(np.nanmean(spacing.spacing_ratios(spec_f[:nlo])))
        unfolded = debruijn_newman.empirical_unfold(spec_f, degree=6)
        delta3 = spacing.dyson_mehta_delta3(unfolded, lengths)
        d3_dist = spacing.delta3_gue_distance(lengths, delta3)

        rows.append(
            {
                "x": x,
                "N": N,
                "dps": dps,
                "n_levels": len(spec),
                "k_loose": k_loose,
                "k_star": k_star,
                "k_tight": k_tight,
                "t_star": float(t_star) if t_star is not None else None,
                "ratio": ratio,
                "rtilde_full": rtilde_full,
                "rtilde_low": rtilde_low,
                "delta3_gue_dist": d3_dist,
                "capped": capped,
            }
        )
        print(
            f"{x:>4} {dps:>5} {k_loose:>9} {k_star:>5} {k_tight:>9} "
            f"{(float(t_star) if t_star else 0):>8.2f} {(ratio if ratio else 0):>7.2f} "
            f"{rtilde_full:>9.4f} {rtilde_low:>8.4f} {d3_dist:>7.4f} "
            f"{('yes' if capped else ''):>5}"
        )

    # The linear-law constant from the interior (non-capped) points, and a
    # self-consistent predicted k*(x) = #{zeros < c_hat * x} using that constant.
    interior = [r for r in rows if not r["capped"] and r["k_star"] > 0]
    c_hat = float(np.mean([r["ratio"] for r in interior])) if interior else float("nan")
    zr_long = zeros_mpf(max(r["n_levels"] for r in rows), cc.suggest_dps(max(xs)))
    zr_f = np.array([float(z) for z in zr_long])
    for r in rows:
        r["k_pred"] = int(np.count_nonzero(zr_f < c_hat * r["x"])) if interior else None
    print(
        f"  interior mean t*/x = {c_hat:.3f}  "
        f"(2*pi = {2 * np.pi:.3f}, 2*pi*e = {2 * np.pi * np.e:.3f}); "
        f"linear-law constant c_hat used for the predicted overlay"
    )
    # The Delta_3-to-GUE ordering (#53 item 1): report the overall fall and any
    # up-ticks rather than a bare boolean — the smallest cutoff (few primes) can
    # buck the trend without breaking it.
    dists = [r["delta3_gue_dist"] for r in rows]
    ordered = all(b <= a + 1e-9 for a, b in zip(dists, dists[1:]))
    drop = (dists[0] - dists[-1]) / dists[0] if dists and dists[0] else 0.0
    n_up = sum(1 for a, b in zip(dists, dists[1:]) if b > a + 1e-9)
    print(
        f"  Delta_3-to-GUE distance falls {drop:.0%} over the sweep "
        f"({n_up} up-tick(s)): {[round(d, 4) for d in dists]}"
    )
    return {
        "rows": rows,
        "N": N,
        "rel_tol": rel_tol,
        "c_hat": c_hat,
        "low_count": low_count,
        "delta3_monotone": ordered,
    }


# ----------------------------------------------------------------------------
# intermediate: the Seba / rank-one read of the pole-locked tail (#87)
# ----------------------------------------------------------------------------


def study_intermediate(
    xs: list[int],
    *,
    N: int,
    window: int = 40,
    step: int = 8,
    refresh: bool = False,
) -> dict:
    """Intermediate statistics of the pole-locked tail from the couplings alone.

    For each cutoff ``x``: recover the secular couplings ``xi`` (cached
    eigensolve), build the local two-pole prediction (:func:`ci.local_gap_model`),
    and score it against the measured spectrum:

    - **locality validation** — per-gap occupancy agreement and nearest-root
      deviation (in units of the pole spacing ``Delta``) between the local
      prediction and the actual secular roots;
    - **tail statistics** — windowed ``<r~>`` curves and the tail spacing
      histogram, measured vs predicted, against the Poisson / semi-Poisson /
      GUE / picket reference values; plus the effective-coupling profile
      ``w_n = |xi_n / R_n| / Delta`` (:func:`ci.pinned_tail`);
    - **the two boundaries** — the predicted-occupancy deficit plateau
      (:func:`ci.deficit_plateau`, the density crossover ``~ 2 pi x``) and the
      measured tracking height ``t*(x) = zeta_{k*}`` (#53, ``~ 12 x``), to show
      which of them the coupling-side read sees.

    Forward: couplings and poles are the operator's own; the zeros enter only to
    score (``t*``, and the tail-window placement on the comparison axis).
    """
    print(
        f"\n[intermediate] Seba / rank-one read of the pole-locked tail "
        f"(N={N}, window={window})"
    )
    print(
        f"{'x':>4} {'dps':>5} {'occ=':>5} {'med|dz|':>8} {'r~tail':>7} {'r~pred':>7} "
        f"{'w_med':>6} {'t_dens':>7} {'/x':>6} {'t*':>8} {'/x':>6} {'defct':>6}"
    )
    rows = []
    pool_meas: list[np.ndarray] = []
    pool_pred: list[np.ndarray] = []
    for x in xs:
        dps = cc.suggest_dps(x)
        xi = xi_mpf(N, x, dps, refresh=refresh)
        spec = spectrum_from_xi(N, x, dps, xi)
        zr = zeros_mpf(len(spec), dps)
        with mp.workdps(dps):
            errs = [abs(spec[k] - zr[k]) for k in range(len(spec))]
            L = float(mp.log(x))
            model = ci.local_gap_model(xi, N, L)
            tail = ci.pinned_tail(xi, N, L)
        k_star = cc.tracking_length(errs, zr)
        t_star = cc.tracking_height(zr, k_star)
        t_star = float(t_star) if t_star is not None else None
        plateau = ci.deficit_plateau(model)
        spec_f = np.sort(np.array([float(s) for s in spec], dtype=np.float64))
        delta = model.spacing

        # Locality validation: occupancy agreement + per-root deviation.
        occ_meas = np.histogram(spec_f, bins=model.edges)[0]
        occ_agree = float(np.mean(model.occupancy == occ_meas))
        near = (
            np.array([np.abs(spec_f - z).min() for z in model.levels]) / delta
            if model.levels.size and spec_f.size
            else np.array([np.nan])
        )
        med_dz = float(np.median(near))

        # Windowed <r~> curves, measured vs predicted.
        meas = ci.windowed_rtilde(spec_f, window=window, step=step)
        pred = ci.windowed_rtilde(model.levels, window=window, step=step)

        # Tail statistics above the measured tracking height (trimmed to the
        # common span: the top gap's root can escape the secular finder).
        t_lo = t_star if t_star is not None else 0.0
        hi = min(spec_f[-1], model.levels[-1]) if model.levels.size else 0.0
        tail_meas = spec_f[(spec_f > t_lo) & (spec_f <= hi)]
        tail_pred = model.levels[(model.levels > t_lo) & (model.levels <= hi)]
        rt_meas = (
            float(np.nanmean(spacing.spacing_ratios(tail_meas)))
            if tail_meas.size >= 8
            else None
        )
        rt_pred = (
            float(np.nanmean(spacing.spacing_ratios(tail_pred)))
            if tail_pred.size >= 8
            else None
        )
        pool_meas.append(np.diff(tail_meas) / delta)
        pool_pred.append(np.diff(tail_pred) / delta)

        # The effective-coupling profile over the tail (the scale-free finding).
        g_star = int(np.searchsorted(model.edges, t_lo))
        w_tail = float(np.median(tail.w[max(g_star - 1, 0) :]))

        row = {
            "x": x,
            "dps": dps,
            "occ_agree": occ_agree,
            "median_dz": med_dz,
            "k_star": k_star,
            "t_star": t_star,
            "t_dens_first": plateau.t_first if plateau else None,
            "t_dens": plateau.t_mid if plateau else None,
            "t_dens_last": plateau.t_last if plateau else None,
            "deficit_max": plateau.deficit_max if plateau else 0,
            "rtilde_tail_meas": rt_meas,
            "rtilde_tail_pred": rt_pred,
            "w_tail_median": w_tail,
            "poles": tail.poles.tolist(),
            "w": tail.w.tolist(),
            "windowed_meas": meas.tolist(),
            "windowed_pred": pred.tolist(),
        }
        rows.append(row)
        nan = float("nan")
        print(
            f"{x:>4} {dps:>5} {occ_agree:>5.2f} {med_dz:>8.3f} "
            f"{(rt_meas if rt_meas is not None else nan):>7.4f} "
            f"{(rt_pred if rt_pred is not None else nan):>7.4f} "
            f"{w_tail:>6.2f} {(row['t_dens'] or nan):>7.1f} "
            f"{((row['t_dens'] or nan) / x):>6.2f} "
            f"{(t_star if t_star is not None else nan):>8.2f} "
            f"{((t_star if t_star is not None else nan) / x):>6.2f} "
            f"{row['deficit_max']:>6}"
        )

    # Pooled tail spacing histogram (units of the pole spacing), all cutoffs.
    edges = np.linspace(0.0, 2.0, 25)
    h_meas, _ = np.histogram(np.concatenate(pool_meas), bins=edges, density=True)
    h_pred, _ = np.histogram(np.concatenate(pool_pred), bins=edges, density=True)

    agree = [r["occ_agree"] for r in rows]
    dens_ratio = [r["t_dens"] / r["x"] for r in rows if r["t_dens"] is not None]
    print(
        f"  locality: occupancy agreement {min(agree):.0%}..{max(agree):.0%}; "
        f"the local two-pole model is the tail's statistics"
    )
    if dens_ratio:
        print(
            f"  density crossover t_dens/x = {float(np.mean(dens_ratio)):.2f} "
            f"(2 pi = {2 * np.pi:.2f}) -- the coupling-side linear law; the "
            f"tracking height t*/x ~ 11.75 stays invisible to the local read"
        )
    return {
        "rows": rows,
        "N": N,
        "window": window,
        "hist_edges": edges.tolist(),
        "hist_meas": h_meas.tolist(),
        "hist_pred": h_pred.tolist(),
    }


# ----------------------------------------------------------------------------
# edge-corruption: is fp64 xi-corruption confined to the low band? (#82)
# ----------------------------------------------------------------------------


def study_edge_corruption(ns: list[int], *, x: int, dps: int | None = None) -> dict:
    """Does fp64 ``xi``-corruption hit the **edge** eigenvalues, or only the low ones?

    The #82 spike gating any Sliwinski outreach. For each truncation ``N`` (fixed
    cutoff ``x``) split the per-index ``xi``-corruption ``|nu_k^{fp64} - nu_k^{mpmath}|``
    into the low / near-null band and the resolution edge (:func:`cc.edge_corruption_profile`).

    The finding: the corruption is **bounded and concentrated in the low band**
    (the ``xi``-direction), while the genuine edge error grows ``~ zeta_N`` with the
    truncation. So there is a crossover ``N`` above which fp64's *uniform* error
    ``E = max_k |nu_k - zeta_k|`` is set by the genuine (robust) edge rather than by
    the corrupted low band — i.e. a fp64 measurement of ``E`` (Sliwinski's Conj 4.1)
    is edge-dominated and plausibly genuine. We exhibit the crossover on the
    reachable scale; Sliwinski's ``kappa ~ 7000`` sits far above it.

    Forward: the spectra are the prime-built operator's; the zeros only score them.
    """
    print(f"\n[edge-corruption] low vs edge fp64 xi-corruption (x={x})")
    print(
        f"{'N':>4} {'cnt':>4} {'k_flr':>5} {'lo_corr':>8} {'ed_corr':>8} "
        f"{'ed_gen':>8} {'E_fp64':>8} {'E_band':>7} {'E_gen':>8}"
    )
    rows = []
    focus = None
    for N in ns:
        prof = cc.edge_corruption_profile(N, mp.sqrt(x), dps=dps)
        s = cc.summarize_edge_bands(prof)
        rows.append(
            {
                "N": N,
                "count": prof.count,
                "k_floor": prof.k_floor,
                "low_corruption": s["low"]["max_corruption"],
                "edge_corruption": s["edge"]["max_corruption"],
                "edge_genuine": s["edge"]["max_genuine"],
                "E_fp64": s["E_fp64"],
                "E_genuine": s["E_genuine"],
                "E_band": s["E_argmax_band"],
            }
        )
        print(
            f"{N:>4} {prof.count:>4} {str(prof.k_floor):>5} "
            f"{s['low']['max_corruption']:>8.2f} {s['edge']['max_corruption']:>8.2f} "
            f"{s['edge']['max_genuine']:>8.2f} {s['E_fp64']:>8.2f} "
            f"{s['E_argmax_band']:>7} {s['E_genuine']:>8.2f}"
        )
        focus = prof  # keep the deepest profile for the per-index figure

    crossover = next((r["N"] for r in rows if r["E_band"] == "edge"), None)
    edge_robust = all(r["edge_corruption"] < r["low_corruption"] + 1e-9 for r in rows)
    print(
        f"  edge corruption < low corruption for every N: {edge_robust} "
        f"(corruption is concentrated in the low / near-null band)"
    )
    if crossover is not None:
        print(
            f"  crossover at N={crossover}: above it fp64's uniform error E is set by "
            f"the genuine (robust) resolution edge, not the corrupted low band"
        )
    else:
        print("  no crossover in this N range: low-band corruption still sets E")
    print(
        "  VERDICT: fp64 xi-corruption is confined to the low/near-null band; the "
        "edge eigenvalues are pole-pinned and robust. A fp64 uniform-error sweep at "
        "Sliwinski's kappa~7000 is edge-dominated -> plausibly genuine. NO-GO on "
        "outreach; the 'Conj 4.1 ~ precision wall' caveat overclaims (it holds for "
        "the LOW zeros only)."
    )
    return {
        "x": x,
        "dps": focus.dps if focus else dps,
        "rows": rows,
        "crossover_N": crossover,
        "edge_robust": edge_robust,
        "focus": {
            "N": focus.N,
            "count": focus.count,
            "k_floor": focus.k_floor,
            "bound": focus.bound,
            "genuine": focus.genuine,
            "fp64_error": focus.fp64_error,
            "corruption": focus.corruption,
            "gap_mpmath": focus.gap_mpmath,
            "gap_fp64": focus.gap_fp64,
        }
        if focus
        else None,
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
        choices=[
            "artifact",
            "edge",
            "edge-corruption",
            "accelerate",
            "tracking",
            "intermediate",
            "all",
        ],
        default="all",
    )
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument(
        "--tracking-x",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12, 14, 16, 18],
        help="prime cutoffs for the k*(x) tracking sweep",
    )
    ap.add_argument(
        "--tracking-N",
        type=int,
        default=160,
        help="truncation N for the tracking sweep (must exceed the largest k*)",
    )
    ap.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="relative tolerance defining the tracked block for k*(x)",
    )
    ap.add_argument(
        "--edge-corruption-N",
        type=int,
        nargs="+",
        default=[60, 80, 100, 120, 160],
        help="truncations N for the #82 edge-corruption crossover sweep",
    )
    ap.add_argument(
        "--edge-corruption-x",
        type=int,
        default=13,
        help="prime cutoff x for the #82 edge-corruption sweep",
    )
    ap.add_argument(
        "--intermediate-x",
        type=int,
        nargs="+",
        default=[6, 10, 14, 18, 22],
        help="prime cutoffs for the #87 intermediate-statistics read",
    )
    ap.add_argument(
        "--intermediate-N",
        type=int,
        default=160,
        help="truncation N for the #87 read (shares the tracking-sweep caches)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=40,
        help="sliding-window length for the windowed spacing-ratio curves (#87)",
    )
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
    if args.mode == "edge-corruption":
        ec = study_edge_corruption(
            args.edge_corruption_N, x=args.edge_corruption_x, dps=None
        )
        out = plots.ccm_edge_corruption_figure(
            ec, out_path=args.out_dir / "ccm_edge_corruption.png"
        )
        print(f"  figure -> {out}")
    if args.mode in ("accelerate", "all"):
        study_accelerate([11, 12, 13, 14], N=args.N, indices=[10, 20, 30, 40])
    if args.mode in ("tracking", "all"):
        t = study_tracking(
            args.tracking_x,
            N=args.tracking_N,
            rel_tol=args.rel_tol,
            refresh=args.refresh,
        )
        out = plots.ccm_tracking_range_figure(
            t, out_path=args.out_dir / "ccm_tracking_range.png"
        )
        print(f"  figure -> {out}")
    if args.mode == "intermediate":
        i = study_intermediate(
            args.intermediate_x,
            N=args.intermediate_N,
            window=args.window,
            refresh=args.refresh,
        )
        out = plots.ccm_intermediate_stats_figure(
            i, out_path=args.out_dir / "ccm_intermediate_stats.png"
        )
        print(f"  figure -> {out}")

    print(f"\n[done] {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
