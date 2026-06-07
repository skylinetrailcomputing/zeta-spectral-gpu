"""Invariants for the forward family GRH Li-criterion probe (issue #71).

The decisive checks pin the forward family computation independently: the assembled
``lambda_n(chi)`` (built from the generalized-Stieltjes constants
``gamma_n(r/q)``) are anchored against an independent **Cauchy/Taylor** pass over the
completed ``log Lambda(1+u, chi)`` -- a different forward route -- the real-vs-complex
character distinction is checked (real characters give real ``lambda_n``; complex
characters genuinely complex, with the verdict on ``Re lambda_n``), and the structural
forward guarantee is asserted (no zero is ever read). GPU tests skip cleanly when cupy
isn't installed and reproduce the mpmath reference on small ``n``.

Small ``n_max`` / modest ``dps`` keep them CI-friendly (the Cauchy/Taylor anchor is the
slow part, so it is used only on tiny inputs).
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from zeta_spectral_gpu import li_criterion as li
from zeta_spectral_gpu import li_criterion_family as fam
from zeta_spectral_gpu.dirichlet import dirichlet_character
from zeta_spectral_gpu.katz_sarnak import quadratic_character


def _cauchy_lambda(char: np.ndarray, n_max: int, dps: int, radius: str = "0.4") -> list:
    """Independent forward ``lambda_n(chi)`` via a Cauchy/Taylor pass over log Lambda.

    A genuinely different route from the module's generalized-Stieltjes producer: form
    ``log Lambda(1+u, chi)`` directly (``log(q/pi)`` + ``loggamma`` + ``log L`` via
    ``mpmath.dirichlet``) and Taylor it numerically, then run the shared
    Bombieri-Lagarias combination. Forward throughout.
    """
    char = np.asarray(char)
    q = char.size
    a = fam.character_parity(char)
    chi_list = [mp.mpc(complex(c)) for c in char]
    with mp.workdps(dps):

        def log_lambda(s):
            return (
                ((s + a) / 2) * mp.log(mp.mpf(q) / mp.pi)
                + mp.loggamma((s + a) / 2)
                + mp.log(mp.dirichlet(s, chi_list))
            )

        coeffs = mp.taylor(log_lambda, 1, n_max, method="quad", radius=mp.mpf(radius))
        return li.li_from_log_coefficients(coeffs, n_max)


def test_parity_even_and_odd():
    # disc 5 > 0 -> even (a=0); disc -3 < 0 and mod-4 odd char -> odd (a=1).
    assert fam.character_parity(quadratic_character(5)) == 0
    assert fam.character_parity(quadratic_character(-3)) == 1
    assert fam.character_parity(dirichlet_character(4, 1)) == 1


def test_real_character_lambda_is_real_and_anchored():
    # Exact Kronecker character (values 0, +-1) -> agreement with the independent
    # Cauchy route, and a vanishing imaginary residual (lambda real).
    char = quadratic_character(5)  # q = 5, even, real
    n_max, dps = 5, 25
    ours = fam.character_li_coefficients(char, n_max, dps=dps)
    indep = _cauchy_lambda(char, n_max, dps)
    assert max(abs(ours[i] - indep[i]) for i in range(n_max)) < mp.mpf("1e-12")
    assert max(abs(mp.im(c)) for c in ours) < mp.mpf("1e-18")
    assert all(mp.re(c) > 0 for c in ours)  # GRH-consistent over the range


def test_complex_character_is_complex_and_anchored():
    # A genuinely complex character (mod 5, j=1): lambda_n is complex, the verdict is
    # on Re lambda_n (> 0), matched to the independent route at fp64 (the np.exp
    # character values are float64 -- the documented precision ceiling).
    char = dirichlet_character(5, 1)
    n_max, dps = 5, 25
    ours = fam.character_li_coefficients(char, n_max, dps=dps)
    indep = _cauchy_lambda(char, n_max, dps)
    assert max(abs(ours[i] - indep[i]) for i in range(n_max)) < mp.mpf("1e-8")
    assert max(abs(mp.im(c)) for c in ours) > 0.01  # genuinely complex
    assert all(mp.re(c) > 0 for c in ours)


def test_l1_equals_known_value():
    # c_0 = L(1, chi); for the odd mod-4 character L(1, chi) = pi/4 (Leibniz / beta(1)).
    char = dirichlet_character(4, 1)
    with mp.workdps(30):  # so mp.pi in the comparison is high-precision too
        c0 = fam.l_taylor_coefficients(char, 1, dps=30)[0]
        assert abs(c0 - mp.pi / 4) < mp.mpf("1e-25")


def test_evaluate_character_reports_verdict():
    res = fam.evaluate_character(quadratic_character(-3), 8, label="chi_-3")
    assert res.rh_consistent and res.all_re_positive
    assert res.is_real and res.parity == 1
    assert res.min_re > 0
    assert res.imag_residual < mp.mpf("1e-15")  # real character
    assert res.stability < mp.mpf("1e-15")  # well-resolved, not cancellation noise


def test_family_verdict_over_quadratic_family():
    chars = fam.quadratic_family(12)
    res = fam.evaluate_family(chars, 6, kind="quadratic")
    assert res.n_members == len(chars) > 3
    assert res.all_positive and res.rh_consistent  # GRH-consistent across the family
    assert res.worst_member.min_re > 0
    assert res.max_imag_residual < mp.mpf("1e-14")  # every member real


def test_family_enumerators():
    # Quadratic family: fundamental discriminants, all real characters.
    quad = fam.quadratic_family(7)
    assert all(fam.is_real_character(c) for _, c in quad)
    # Prime family: non-principal characters of prime modulus, including complex.
    prime = fam.prime_family(7)
    labels = [lab for lab, _ in prime]
    assert "chi_5.1" in labels and "chi_7.3" in labels  # mod 5 and mod 7 present
    assert any(not fam.is_real_character(c) for _, c in prime)  # complex characters
    assert "chi_2.0" not in labels  # mod 2 / principal excluded
    # "all" == quadratic ++ prime (compare labels; arrays don't support == cleanly).
    all_labels = [lab for lab, _ in fam.dirichlet_family("all", 7)]
    assert all_labels == [lab for lab, _ in quad] + labels


def test_prime_family_complex_character_positive():
    # The complex side of the family is GRH-consistent (Re lambda_n > 0).
    res = fam.evaluate_family(fam.prime_family(7), 6, kind="prime")
    assert res.all_positive
    assert any(not m.is_real for m in res.members)  # complex members present


def test_forward_no_zeros_consumed(monkeypatch):
    # Structural guarantee: the family producer must not read a single zero. Poison
    # mpmath.zetazero so any inverse-style access would explode.
    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            "li_criterion_family must not consume zeros (forward only)"
        )

    monkeypatch.setattr(mp, "zetazero", _boom)
    coeffs = fam.character_li_coefficients(quadratic_character(5), 5, dps=30)
    assert all(mp.re(c) > 0 for c in coeffs)


# --- GPU agreement (the house GPU-vs-CPU rule) --------------------------------


def test_gpu_matches_cpu_family():
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import li_criterion_family_gpu as gpu

    chars = fam.quadratic_family(12) + fam.prime_family(7)
    n_max, dps = 10, 40
    lam, min_re, all_pos = gpu.family_li_coefficients_gpu(chars, n_max, dps=dps)
    assert lam.shape == (len(chars), n_max)
    for c, (_, char) in enumerate(chars):
        ref = fam.character_li_coefficients(char, n_max, dps=dps)
        ref_c = np.array([complex(z) for z in ref])
        np.testing.assert_allclose(lam[c], ref_c, rtol=0, atol=1e-7)


def test_gpu_verdict_matches_cpu():
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import li_criterion_family_gpu as gpu

    chars = fam.quadratic_family(12)
    n_max = 10
    _, min_re, all_pos = gpu.family_li_coefficients_gpu(chars, n_max, dps=40)
    for c, (_, char) in enumerate(chars):
        res = fam.evaluate_character(char, n_max, dps=40)
        assert bool(all_pos[c]) == res.all_re_positive
        assert abs(min_re[c] - float(res.min_re)) < 1e-7
