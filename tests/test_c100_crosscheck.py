"""Invariants for the independent ``c=100`` cross-check fixture (#94 follow-up).

``tests/fixtures/c100_crosscheck.json`` records *our* forward ``gamma_1..gamma_10``
at Groskin's published ``c=100`` extraction cells (``N in {120,150,160,250}``;
``N=120/160`` are the cells he ran at our grid points), produced by
``zeta_spectral_gpu.ccm`` with a closed-form (``T=inf``) archimedean term. The always-on tests here guard the fixture and its honesty using only
``mpmath`` (a core dep), so they run in CI. The heavy rebuild — re-running our
assembly and matching the fixture to hundreds of digits — is a ``slow`` test,
deselected by default.

Forward: the recorded ``abs_error`` is ``|gamma_k - t_k|``; ``t_k`` is only ever a
yardstick, never an input.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from zeta_spectral_gpu import ccm, oracle

# First nontrivial ordinate t_1 to ample precision, independent of the fixture and
# of mp.zetazero — the honesty anchor (cf. test_connes_cvs_oracle.GAMMA1).
GAMMA1 = (
    "14.1347251417346937904572519835624702707842571156992431756855674601499"
    "63429809256764949010393171561012779202971548797436766142691469882254582"
    "5053632394471377804133812372059705496219558658602"
)


def test_fixture_structure():
    rec = oracle.load_c100_crosscheck()
    cells = rec["cells"]
    assert [(c["cell"]["N"], c["cell"]["dps"]) for c in cells] == [
        (120, 560),
        (150, 1000),
        (160, 560),
        (250, 500),
    ]
    for c in cells:
        assert c["cell"]["c"] == 100
        assert c["cell"]["T"] == "inf (closed-form)"
        assert len(c["gamma"]) == 10
        for g in c["gamma"]:
            assert {"k", "gamma", "stable_digits", "abs_error"} <= set(g)


def test_errors_grow_with_index():
    # Higher zeros resolve less deeply, so |gamma_k - t_k| increases strictly in k
    # within each cell — a forward sanity invariant of the finite-cutoff spectrum.
    # Compared as mpf (the errors run to ~1e-330, below float64's range).
    with mp.workdps(40):
        for c in oracle.load_c100_crosscheck()["cells"]:
            errs = [mp.mpf(g["abs_error"]) for g in c["gamma"]]
            assert all(b > a for a, b in zip(errs, errs[1:]))


def test_deeper_N_resolves_deeper():
    # N=250 must track the first zero far deeper than N=150 (finite-N is the floor).
    cells = {c["cell"]["N"]: c for c in oracle.load_c100_crosscheck()["cells"]}
    with mp.workdps(40):
        e250 = mp.mpf(cells[250]["gamma"][0]["abs_error"])
        e150 = mp.mpf(cells[150]["gamma"][0]["abs_error"])
    assert e250 < e150


def test_published_digits_are_certified():
    # We never publish more digits than the +pad-dps confirmation run agreed on.
    for c in oracle.load_c100_crosscheck()["cells"]:
        for g in c["gamma"]:
            mantissa = g["gamma"].lstrip("-").replace(".", "").split("e")[0]
            assert len(mantissa.rstrip("0")) <= g["stable_digits"] + 1


def test_gamma1_is_honest_against_known_ordinate():
    # Independent check: our recorded gamma_1 agrees with the true t_1 to about the
    # number of digits the recorded abs_error implies, using a hardcoded t_1 (no
    # operator ever sees a zero). For N=250, abs_error ~ 5e-330, so the two must
    # agree well past the ~120 digits of GAMMA1 we anchor against.
    cells = {c["cell"]["N"]: c for c in oracle.load_c100_crosscheck()["cells"]}
    with mp.workdps(160):
        t1 = mp.mpf(GAMMA1)
        for N in (150, 250):
            g1 = mp.mpf(cells[N]["gamma"][0]["gamma"])
            assert abs(g1 - t1) / abs(t1) < mp.mpf("1e-120")


@pytest.mark.slow
def test_crosscheck_reproduces_fixture():
    # Heavy: rebuild gamma_1 at the N=250 cell from our assembly and confirm it
    # matches the recorded value to many digits. ~70s at dps=500.
    rec = oracle.load_c100_crosscheck()
    cell = next(c for c in rec["cells"] if c["cell"]["N"] == 250)
    dps = cell["cell"]["dps"]
    res = ccm.converge(250, mp.sqrt(100), 1, dps=dps)
    with mp.workdps(dps + 20):
        recorded = mp.mpf(cell["gamma"][0]["gamma"])
        assert abs(res.eigenvalues[0] - recorded) / abs(recorded) < mp.mpf("1e-300")
