"""Baseline oracle (#16): reproduce the connes-cvs c=13 A/B cell as a known-good
``lambda_min`` and record it as a checked-in fixture.

``connes-cvs`` (Groskin 2026, PyPI, MIT) is the first public implementation of the
Connes–van Suijlekom Galerkin matrix ``Q(c)``. Its ground-state eigenvalue
``lambda_min(c)`` measures how close the zeta zeros come to Weil positivity; as the
prime cutoff ``c`` grows, ``lambda_min(c) -> 0`` super-exponentially and the
eigenvector encodes the low zeros.

We use it **as an oracle / dev-dependency only** — never a runtime dependency. It
is itself forward (it builds ``Q(c)`` from the prime / von-Mangoldt sum and
extracts zeros as *output*); here we merely cross-check our own forward numbers
against it, and we never fit to it. This stands up the validation target before we
write any of our own assembly (#8 CPU multiprecision reference, #9 GPU assembly).

The documented A/B cell is ``c=13, N=80, T=400, dps=80`` with the known-good
target ``lambda_min ≈ 2.528266…e-59``. This is a CPU/mpmath computation: the
quantity lives at ~1e-59, far below fp64, so there is nothing to put on the GPU
here.

    uv sync --extra oracle
    uv run python scripts/run_connes_cvs_oracle.py            # verify only
    uv run python scripts/run_connes_cvs_oracle.py --write    # (re)write fixture
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "connes_cvs_c13_oracle.json"
)

# Known-good target for the c=13 A/B cell (Groskin 2026 / seed doc). Sanity bound
# only — the recorded fixture carries the full-precision value the run produces.
TARGET_LAMBDA_MIN = mp.mpf("2.528266e-59")

# Published first-zero errors |gamma_1 - t_1| at c=13, for the normalization note.
# These are *not* lambda_min; they are the zero-error the papers tabulate. The
# ratio of connes-cvs's |gamma_1 err| to the published values is the ~1.2-1.3
# normalization-convention factor (#16 deliverable) — different N/T/precision and
# normalization conventions, not a correctness gap. See knowledge/connes-cvs-oracle.md.
PUBLISHED_GAMMA1_ERR = {
    "ccm_2025_sec6": "2.44e-55",  # N=120, 200 dps (arXiv:2511.22755 §6)
    "connes_2026": "2.6e-55",  # first-50 data at c=13 (arXiv:2602.04022)
}


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def compute_cell(c: int, n: int, t: int, dps: int) -> dict:
    """Run the connes-cvs A/B cell and return a JSON-ready provenance record."""
    import connes_cvs as cc

    mp.mp.dps = dps
    q = cc.build_galerkin_matrix(c=c, N=n, T=t, dps=dps)
    lambda_min, eigvec = cc.compute_ground_state(q)
    zeros = cc.extract_zeros(eigvec, L=mp.log(c), n_zeros=1, dps=dps)
    z0 = zeros[0] if zeros else {}

    # Stringify every mp value at full precision; floats cannot hold ~1e-59.
    def s(x) -> str:
        return mp.nstr(mp.mpf(x), dps, strip_zeros=False)

    detected = z0.get("gamma_detected")
    err = z0.get("error")
    return {
        "_about": (
            "Known-good oracle for the Connes-van Suijlekom Galerkin matrix Q(c). "
            "connes-cvs is used as a forward cross-check / dev-dependency only; we "
            "never fit to it. Regenerate with scripts/run_connes_cvs_oracle.py --write."
        ),
        "source": {
            "package": "connes-cvs",
            "package_version": _pkg_version("connes-cvs"),
            "python_flint_version": _pkg_version("python-flint"),
            "doi": "10.5281/zenodo.19546514",
            "repo": "https://github.com/akivag613/connes-cvs-",
        },
        "cell": {"c": c, "N": n, "T": t, "dps": dps},
        "lambda_min": s(lambda_min),
        "lambda_min_float": float(lambda_min),
        "gamma1_detected": s(detected) if detected is not None else None,
        "gamma1_abs_error": s(err) if err is not None else None,
        "gamma1_abs_error_float": float(err) if err is not None else None,
        "normalization_note": {
            "what": (
                "lambda_min is connes-cvs's internal ground-state eigenvalue. The "
                "CCM/Connes papers tabulate the *zero error* |gamma_k - t_k|, not "
                "lambda_min. The ~1.2-1.3 factor below relates connes-cvs's first-zero "
                "error to the published values; it reflects N/T/precision and "
                "normalization conventions, not a correctness gap. Interpret absolute "
                "lambda_min with this in mind."
            ),
            "published_gamma1_err": PUBLISHED_GAMMA1_ERR,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--c", type=int, default=13)
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--dps", type=int, default=80)
    ap.add_argument("--write", action="store_true", help="write/update the fixture")
    args = ap.parse_args()

    print(f"connes-cvs oracle: c={args.c}, N={args.N}, T={args.T}, dps={args.dps}")
    rec = compute_cell(args.c, args.N, args.T, args.dps)

    lam = mp.mpf(rec["lambda_min"])
    rel = abs(lam - TARGET_LAMBDA_MIN) / TARGET_LAMBDA_MIN
    print(f"  lambda_min      = {rec['lambda_min']}")
    print(f"  target (~)      = {mp.nstr(TARGET_LAMBDA_MIN, 7)}")
    print(f"  |rel - target|  = {mp.nstr(rel, 3)}  (sanity; expect << 1)")
    print(f"  gamma1 detected = {rec['gamma1_detected']}")
    print(f"  |gamma1 error|  = {rec['gamma1_abs_error']}")

    if rel > mp.mpf("1e-3"):
        raise SystemExit(
            f"lambda_min {rec['lambda_min']} is not within 0.1% of the documented "
            f"target {mp.nstr(TARGET_LAMBDA_MIN, 7)} — investigate before recording."
        )
    print("  OK: matches the documented c=13 A/B cell.")

    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote fixture -> {FIXTURE}")


if __name__ == "__main__":
    main()
