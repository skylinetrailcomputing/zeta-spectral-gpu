"""Known-good oracle values for the flagship CCM/CvS computation (#16).

A thin loader over the checked-in fixture ``tests/fixtures/connes_cvs_c13_oracle.json``,
which records the ground-state eigenvalue ``lambda_min`` of the Connes–van
Suijlekom Galerkin matrix ``Q(c)`` at the documented ``c=13, N=80, T=400, dps=80``
cell, as produced by ``connes-cvs`` (Groskin 2026).

The fixture stores ``lambda_min`` and the first-zero error as full-precision
decimal *strings* (the values live at ~1e-59 / ~1e-55, far below fp64). These
helpers parse them back as ``mpmath.mpf`` at enough precision to be lossless, so
the upcoming CPU reference (#8) and GPU assembly (#9) can diff against a stable
target.

``connes-cvs`` is a forward construction used here as a cross-check only — see
``knowledge/connes-cvs-oracle.md``. Loading this fixture does **not** require
``connes-cvs`` to be installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

ORACLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "connes_cvs_c13_oracle.json"
)

C100_CROSSCHECK_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "c100_crosscheck.json"
)


def load_connes_cvs_oracle() -> dict:
    """Return the raw oracle record (cell params, lambda_min, provenance)."""
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


def load_c100_crosscheck() -> dict:
    """Return our independent ``c=100`` results (``gamma_1..gamma_10`` per cell).

    These are *our* forward numbers — produced by ``zeta_spectral_gpu.ccm`` with a
    closed-form (``T=inf``) archimedean term — recorded so they can be diffed
    against Groskin's ``connes-cvs`` ``data/c100/`` files. Regenerate with
    ``scripts/run_c100_crosscheck.py --write``. Loading needs no extra deps.
    """
    return json.loads(C100_CROSSCHECK_PATH.read_text(encoding="utf-8"))


def _as_mpf(value: str, record: dict, dps: int | None) -> mp.mpf:
    # Parse at >= the precision the value was recorded with, so no digits are
    # silently dropped by a lower ambient mp.mp.dps.
    target = dps if dps is not None else int(record["cell"]["dps"]) + 5
    with mp.workdps(target):
        return mp.mpf(value)


def lambda_min(dps: int | None = None) -> mp.mpf:
    """The recorded ground-state eigenvalue ``lambda_min(c=13)`` as an ``mpf``."""
    rec = load_connes_cvs_oracle()
    return _as_mpf(rec["lambda_min"], rec, dps)


def gamma1_abs_error(dps: int | None = None) -> mp.mpf:
    """The recorded first-zero error ``|gamma_1 - t_1|`` as an ``mpf``."""
    rec = load_connes_cvs_oracle()
    return _as_mpf(rec["gamma1_abs_error"], rec, dps)
