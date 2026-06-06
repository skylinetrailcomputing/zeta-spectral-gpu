"""Warm-up spike #20: De Bruijn–Newman forward flow — does increasing ``t`` raise
the zeros' rigidity?

Generates the real zeros of ``H_t(z) = integral_0^inf e^{t u^2} Phi(u) cos(z u) du``
*directly from the structural kernel* ``Phi`` (Polymath15, arXiv:1904.12438) for
several ``t >= 0``, unfolds each set by its own smooth count, and measures
rigidity with the #15 statistics (``Sigma^2(L)`` and ``Delta_3(L)``) plus the
nearest-neighbour spacing variance.

The De Bruijn–Newman constant ``Lambda`` is defined by: ``H_t`` has only real
zeros iff ``t >= Lambda`` (Rodgers–Tao: ``Lambda >= 0``; Polymath15:
``Lambda <= 0.22``; RH <=> ``Lambda <= 0``). For ``t >= 0`` the heat flow keeps
the zeros real and makes them progressively *more rigid* — toward an evenly
spaced picket fence as ``t -> inf``.

Success criterion (the go/no-go for full 7b): rigidity rises with ``t`` —
``Sigma^2(L)`` and ``Delta_3(L)`` fall and the spacing variance shrinks as ``t``
increases. A clear qualitative trend is a *go*; intractable ``H_t``-zero numerics
at useful height would be a documented *no-go*.

Forward check: the zeros are only an *output* being characterised — the known
zeta zeros are never fed in. ``H_t`` is the structural kernel's cosine transform,
and we root-find it (we do *not* flow the known zeros as particles).

    uv run python scripts/run_debruijn_newman.py
    uv run python scripts/run_debruijn_newman.py --n 50 --t 0 0.1 0.5 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from zeta_spectral_gpu import debruijn_newman as dbn
from zeta_spectral_gpu import plots, spacing

DATA = Path(__file__).resolve().parent.parent / "data"


def _verify_t0_against_ordinates(z0: np.ndarray, k: int = 5) -> float:
    """Max |forward H_0 root - 2*gamma| over the first ``k`` true ordinates."""
    gammas = np.array([float(mp.zetazero(j).imag) for j in range(1, k + 1)])
    return float(np.max(np.abs(z0[:k] - 2.0 * gammas)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=40, help="zeros generated per t")
    ap.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.5, 1.0, 2.0],
        help="De Bruijn–Newman times to sweep",
    )
    ap.add_argument("--dps", type=int, default=None, help="mpmath precision override")
    ap.add_argument("--scan-step", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=6, help="unfolding polynomial degree")
    ap.add_argument("--l-min", type=float, default=0.5)
    ap.add_argument("--l-max", type=float, default=8.0)
    ap.add_argument("--n-l", type=int, default=18)
    ap.add_argument("--n-offsets", type=int, default=600)
    ap.add_argument("--l-ref", type=float, default=3.0, help="L for the trend table")
    ap.add_argument("--out", type=Path, default=DATA / "debruijn_newman.png")
    ap.add_argument("--refresh", action="store_true", help="ignore cached zeros")
    args = ap.parse_args()

    lengths = np.geomspace(args.l_min, args.l_max, args.n_l)
    sigma2_by_t: dict[float, np.ndarray] = {}
    delta3_by_t: dict[float, np.ndarray] = {}
    rows = []
    z0: np.ndarray | None = None

    print(f"De Bruijn–Newman forward flow  (n={args.n} zeros/t, degree={args.degree})")
    for t in args.t:
        t0 = time.perf_counter()
        z = dbn.h_t_zeros(
            t, args.n, dps=args.dps, scan_step=args.scan_step, refresh=args.refresh
        )
        if t == 0.0:
            z0 = z
        w = dbn.empirical_unfold(z, degree=args.degree)
        s = np.diff(w)
        sigma2 = spacing.number_variance(w, lengths, n_offsets=args.n_offsets)
        delta3 = spacing.dyson_mehta_delta3(w, lengths, n_offsets=args.n_offsets)
        sigma2_by_t[float(t)] = sigma2
        delta3_by_t[float(t)] = delta3
        s2_ref = float(np.interp(args.l_ref, lengths, sigma2))
        d3_ref = float(np.interp(args.l_ref, lengths, delta3))
        rows.append((t, z.size, s.var(), s2_ref, d3_ref))
        print(f"  t={t:4.1f}  {z.size:2d} zeros  [{time.perf_counter() - t0:5.0f}s]")

    if z0 is not None:
        err = _verify_t0_against_ordinates(z0)
        print(f"\nforward check: max|H_0 root - 2*gamma| (first 5) = {err:.2e}")

    print(f"\nrigidity vs t  (L_ref={args.l_ref:g}; falling = more rigid):")
    print("    t    zeros   var(s)   Sigma^2   Delta_3")
    for t, nz, var_s, s2, d3 in rows:
        print(f"  {t:4.1f}   {nz:4d}   {var_s:6.3f}   {s2:7.4f}   {d3:7.4f}")
    print(
        f"  GUE ref at L={args.l_ref:g}:                "
        f"{float(spacing.gue_number_variance(args.l_ref)):7.4f}   "
        f"{float(spacing.gue_delta3(args.l_ref)):7.4f}"
    )

    s2_series = [r[3] for r in rows]
    var_series = [r[2] for r in rows]
    monotone = all(b <= a + 1e-9 for a, b in zip(s2_series, s2_series[1:]))
    verdict = "GO" if (monotone or var_series[-1] < var_series[0]) else "INCONCLUSIVE"
    print(
        f"\nverdict: rigidity {'rises' if verdict == 'GO' else 'does NOT clearly rise'}"
        f" with t  ->  {verdict} for full 7b"
    )

    out = plots.dbn_rigidity_figure(
        lengths,
        sigma2_by_t,
        delta3_by_t,
        out_path=args.out,
        title=f"De Bruijn–Newman flow: rigidity increases with $t$  (n={args.n}/t)",
    )
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
