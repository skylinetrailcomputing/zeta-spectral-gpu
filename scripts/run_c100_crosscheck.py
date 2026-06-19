"""Independent ``c=100`` cross-check vs ``connes-cvs`` (#94 follow-up).

This is the **our-side** companion to ``run_connes_cvs_oracle.py`` (which runs
*Groskin's* ``connes-cvs`` package). Here we run *our own* forward assembly —
``zeta_spectral_gpu.ccm`` — at Groskin's published ``c=100`` cells and record the
first ten eigenvalues ``gamma_1 .. gamma_10`` to many digits, so anyone (Akiva
included) can diff our numbers against his ``data/c100/`` files line by line.

Forward, not inverse: the eigenvalues come from the prime-built operator; the true
ordinates ``t_k = Im(zeta zero_k)`` enter only to *score* them (the ``abs_error``
column), never as input. Our archimedean term is **closed-form** (the digamma
divided-difference ``q_{m,n} = (psi(m)-psi(n))/(m-n)``, ``q_{n,n} = psi'(n)``), so
there is **no ``T``-quadrature** on our path — matching a cell needs only ``N`` and
an adequate ``dps``. The gap between our ``gamma_k`` and Groskin's ``gamma_detected``
(same ``N``, his ``T=800`` quadrature) is therefore a clean read on his
``T``-truncation. See ``knowledge/ccm-convergence-law.md``.

Groskin's published cells (akivag613/connes-cvs-, ``data/c100/``):
``N in {100,150,200,250}, T=800, dps=500``. The extraction cells we record our
closed-form ``gamma_k`` at are his two originals — ``N=150,dps=1000`` and
``N=250,dps=500`` — plus ``N=120`` and ``N=160`` (``dps=560``), the two cells he ran
at *our* grid points (issue #1, the finite-``T`` mirror of our ``N=250`` run).

    uv run python scripts/run_c100_crosscheck.py                       # report
    uv run python scripts/run_c100_crosscheck.py --write              # (re)write fixture
    uv run python scripts/run_c100_crosscheck.py --diff PATH_TO_HIS_JSON  # side-by-side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp

from zeta_spectral_gpu import ccm

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "c100_crosscheck.json"
)

# Groskin's published c=100 extraction cells. The first two are his original
# (N=150,dps=1000) and (N=250,dps=500); the latter two (N=120,N=160 at dps=560) are
# the cells he ran at *our* grid points (issue #1, 2026-06-19), the mirror of our
# N=250 run — finite-T against our closed-form. Listed ascending in N.
CELLS = [
    {"c": 100, "N": 120, "dps": 560, "count": 10},
    {"c": 100, "N": 150, "dps": 1000, "count": 10},
    {"c": 100, "N": 160, "dps": 560, "count": 10},
    {"c": 100, "N": 250, "dps": 500, "count": 10},
]

# How many significant digits to publish per gamma_k. We never store more than the
# run is *verified* stable to (see compute_cell): a +pad-dps confirmation run pins
# the trustworthy digit count, and we store the smaller of that and this cap.
PUBLISH_DIGITS = 400
STABILITY_PAD = 80  # extra dps for the confirmation run
STABILITY_GUARD = 10  # digits shaved off the confirmed agreement, for safety


def _matching_digits(a: mp.mpf, b: mp.mpf) -> int:
    if a == b:
        return 10**9
    return int(mp.floor(-mp.log10(abs(a - b) / abs(a))))


def _spectrum(N: int, c: int, count: int, dps: int) -> tuple[list, list, list]:
    """Our forward gamma_k, t_k, |gamma_k - t_k| at one cell."""
    res = ccm.converge(N, mp.sqrt(c), count, dps=dps)
    return res.eigenvalues, res.zeros, res.errors


def compute_cell(c: int, N: int, dps: int, count: int) -> dict:
    """Run our assembly at one cell, verify digit-stability, return a JSON record."""
    eigs, zeros, errors = _spectrum(N, c, count, dps)
    # Confirmation run at higher precision: only digits that survive are published.
    eigs_hi, _, _ = _spectrum(N, c, count, dps + STABILITY_PAD)

    with mp.workdps(dps + STABILITY_PAD):
        gammas = []
        for k in range(count):
            stable = _matching_digits(eigs[k], eigs_hi[k]) - STABILITY_GUARD
            ndig = max(1, min(PUBLISH_DIGITS, stable))
            # No abs_error_float: these run to ~1e-330, far below float64's
            # subnormal floor (~5e-324), so a float would silently underflow to 0.
            gammas.append(
                {
                    "k": k + 1,
                    "gamma": mp.nstr(eigs[k], ndig, strip_zeros=False),
                    "stable_digits": int(stable),
                    "abs_error": mp.nstr(errors[k], 25, strip_zeros=False),
                }
            )
    return {
        "cell": {"c": c, "N": N, "T": "inf (closed-form)", "dps": dps, "count": count},
        "gamma": gammas,
    }


def build_record() -> dict:
    return {
        "_about": (
            "Our independent forward c=100 results: gamma_1..gamma_10 of the "
            "prime-built Connes-van Suijlekom operator (zeta_spectral_gpu.ccm), with a "
            "CLOSED-FORM (T=inf) archimedean term -- no T-quadrature. The zeros enter "
            "only to score (abs_error); never fit. Each gamma is published only to the "
            "digit count a +80-dps confirmation run agrees on. Regenerate: "
            "scripts/run_c100_crosscheck.py --write. Cross-checks vs connes-cvs "
            "data/c100/ (Groskin 2026); see knowledge/ccm-convergence-law.md."
        ),
        "source": {
            "module": "zeta_spectral_gpu.ccm",
            "entry": "converge(N, sqrt(c), count, dps)",
            "archimedean": "closed-form digamma divided-difference (T=inf, no quadrature)",
            "repo": "https://github.com/skylinetrailcomputing/zeta-spectral-gpu",
        },
        "compared_against": {
            "repo": "https://github.com/akivag613/connes-cvs-",
            "data": "data/c100/",
            "T": 800,
        },
        "cells": [compute_cell(**cell) for cell in CELLS],
    }


def diff_against(his_path: str) -> None:
    """Print a per-k side-by-side of our gamma vs Groskin's extraction file."""
    his = json.loads(Path(his_path).read_text(encoding="utf-8"))
    c, N, dps = his["c"], his["N"], his["dps"]
    count = len(his["gamma"])
    print(f"\nDiff vs {his_path}: c={c}, N={N}, his dps={dps}, T=800")
    eigs, zeros, errors = _spectrum(N, c, count, max(dps, 520))
    with mp.workdps(max(dps, 520) + 20):
        print(
            f"  {'k':>2} {'our|g-t|':>14} {'his|g-t|':>14} "
            f"{'agree(our g, his g_det)':>24} {'agree(our g, t)':>16}"
        )
        for hk in his["gamma"]:
            k = hk["k"]
            our_g, our_err = eigs[k - 1], errors[k - 1]
            his_err = mp.mpf(hk["error"])
            if "gamma_detected" in hk:
                his_g = mp.mpf(hk["gamma_detected"])
                a_gdet = f"{_matching_digits(our_g, his_g)} digits"
            else:
                a_gdet = "(no g_det published)"
            t_k = mp.mpf(hk.get("gamma_true") or hk["zetazero_k_imag_reference"])
            a_t = f"{_matching_digits(our_g, t_k)} digits"
            print(
                f"  {k:>2} {mp.nstr(our_err, 6):>14} {mp.nstr(his_err, 6):>14} "
                f"{a_gdet:>24} {a_t:>16}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true", help="write/update the fixture")
    ap.add_argument(
        "--diff", metavar="JSON", help="side-by-side vs a connes-cvs extraction file"
    )
    args = ap.parse_args()

    if args.diff:
        diff_against(args.diff)
        return

    print("Our forward c=100 gamma_1..gamma_10 (closed-form, T=inf):")
    rec = build_record()
    for cell in rec["cells"]:
        cp = cell["cell"]
        print(f"\n  c={cp['c']}, N={cp['N']}, dps={cp['dps']}:")
        for g in cell["gamma"]:
            print(
                f"    gamma_{g['k']:<2} |g-t|={g['abs_error']:>30}  "
                f"(published to {min(PUBLISH_DIGITS, g['stable_digits'])} digits)"
            )

    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote fixture -> {FIXTURE}")


if __name__ == "__main__":
    main()
