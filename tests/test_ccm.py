"""Invariants for the Connes–Consani–Moscovici CPU reference (#8).

The decisive test is ``test_forward_spectrum_matches_ordinates``: a *structurally
derived* operator (built from the primes, never from the zeros) whose spectrum
reproduces the low zeta ordinates — the forward content of the flagship. The rest
are component checks: the prime/von-Mangoldt input, the assembly primitive, the
archimedean closed forms against direct quadrature (the spec's recommended
cross-check), matrix symmetry/parity, and that inverse iteration agrees with a
full ``eigsy``.

All CPU/mpmath at modest precision and small ``N`` so they stay CI-friendly; the
~200-digit reproduction of the source's §6 table lives in ``scripts/run_ccm.py``.
"""

from __future__ import annotations

import mpmath as mp

from zeta_spectral_gpu import ccm, plots


def test_prime_powers_and_von_mangoldt():
    mp.mp.dps = 30
    pp = ccm.prime_powers(13)
    assert [k for k, _ in pp] == [2, 3, 4, 5, 7, 8, 9, 11, 13]
    weights = {k: v for k, v in pp}
    assert abs(weights[4] - mp.log(2)) < 1e-25  # 4 = 2^2  -> Lambda = log 2
    assert abs(weights[9] - mp.log(3)) < 1e-25  # 9 = 3^2  -> Lambda = log 3
    assert abs(weights[7] - mp.log(7)) < 1e-25
    # Cutoff is inclusive of x; composites with two prime factors are excluded.
    assert [k for k, _ in ccm.prime_powers(9)] == [2, 3, 4, 5, 7, 8, 9]
    assert 6 not in [k for k, _ in ccm.prime_powers(14)]  # Lambda(6) = 0


def test_q_kernel_symmetry_and_endpoints():
    mp.mp.dps = 30
    L = 2 * mp.log(mp.sqrt(13))
    # Off-diagonal vanishes at y=0; diagonal equals 2 at y=0.
    assert abs(ccm.q_kernel(2, 5, 0, L)) < 1e-25
    assert abs(ccm.q_kernel(3, 3, 0, L) - 2) < 1e-25
    # Symmetric in (m, n).
    for y in (mp.mpf("0.4"), mp.mpf("1.1")):
        assert abs(ccm.q_kernel(2, 5, y, L) - ccm.q_kernel(5, 2, y, L)) < 1e-25
    # q vanishes at y = L (prime k = lambda^2 contributes nothing).
    assert abs(ccm.q_kernel(2, 5, L, L)) < 1e-25
    assert abs(ccm.q_kernel(4, 4, L, L)) < 1e-25


def test_lerch_matches_mpmath():
    # The direct geometric sum equals mpmath's general Lerch for the small-|z|
    # regime we use (z = e^{-2L}); this is what speeds the fill ~100x.
    mp.mp.dps = 40
    z = mp.exp(-2 * 2 * mp.log(mp.sqrt(13)))
    for s in (1, 2):
        for a in (mp.mpf(1) / 4, mp.mpc(0, 0.7) + mp.mpf(1) / 4):
            assert abs(ccm._lerch(z, s, a) - mp.lerchphi(z, s, a)) < 1e-35


def test_archimedean_closed_form_matches_quadrature():
    # The spec's robust cross-check: the J/K/M closed forms (incl. the delicate
    # diagonal reduction) reproduce direct high-precision quadrature of (4.4).
    mp.mp.dps = 30
    L = 2 * mp.log(mp.sqrt(13))
    for n, m in [(0, 1), (3, 7), (5, 6), (4, 4), (0, 0), (9, 9)]:
        cf = ccm.archimedean_entry(n, m, L)
        qd = ccm.archimedean_entry_quad(n, m, L)
        assert abs(cf - qd) < 1e-25 * (abs(qd) + 1)


def test_archimedean_jkm_fusion_is_bit_identical():
    # The assembly loop uses the fused _archimedean_jkm (one hyp2f1/digamma per
    # mode); it must return *exactly* the standalone archimedean_J/K/M values, since
    # only the redundant special-function calls are removed, not the arithmetic.
    mp.mp.dps = 50
    L = 2 * mp.log(mp.sqrt(13))
    qq = mp.exp(-2 * L)
    exp_mhalf = mp.exp(-L / 2)
    m_const = 2 * exp_mhalf * mp.hyp2f1(mp.mpf(1) / 4, 1, mp.mpf(5) / 4, qq)
    dig_quarter = mp.digamma(mp.mpf(1) / 4)
    for n in (0, 1, 3, 7, 12):
        j, k, m = ccm._archimedean_jkm(
            n, L, qq=qq, exp_mhalf=exp_mhalf, m_const=m_const, dig_quarter=dig_quarter
        )
        assert j == ccm.archimedean_J(n, L)
        assert k == ccm.archimedean_K(n, L)
        assert m == ccm.archimedean_M(n, L)


def test_point_mass_symmetric_and_closed_form():
    mp.mp.dps = 30
    L = 2 * mp.log(mp.sqrt(13))
    assert abs(ccm.point_mass_entry(3, 7, L) - ccm.point_mass_entry(7, 3, L)) < 1e-30
    # Even in the joint sign flip (n,m) -> (-n,-m).
    assert abs(ccm.point_mass_entry(3, 7, L) - ccm.point_mass_entry(-3, -7, L)) < 1e-30


def test_weil_matrix_symmetric_and_parity_even():
    mp.mp.dps = 60
    N = 8
    A = ccm.assemble_weil_matrix(N, mp.sqrt(13))
    assert A.rows == A.cols == 2 * N + 1
    # Exact real symmetry.
    asym = max(abs(A[i, j] - A[j, i]) for i in range(A.rows) for j in range(A.cols))
    assert asym == 0
    # Commutes with the parity grading gamma (V_j -> V_{-j}): A[i,j] = A[2N-i, 2N-j].
    worst = max(
        abs(A[i, j] - A[2 * N - i, 2 * N - j])
        for i in range(A.rows)
        for j in range(A.cols)
    )
    assert worst < mp.mpf(10) ** (-55)


def test_inverse_iteration_matches_eigsy():
    mp.mp.dps = 80
    N = 12
    L = 2 * mp.log(mp.sqrt(13))
    A = ccm.assemble_weil_matrix(N, mp.sqrt(13))
    mode = ccm.smallest_even_eigenvector(A, N)

    E, Q = mp.eigsy(A)
    idx = min(range(len(E)), key=lambda i: abs(E[i]))
    assert abs(mode.eigenvalue - E[idx]) < 1e-40 * (abs(E[idx]) + 1)
    assert mode.parity_residual < mp.mpf(10) ** (-50)

    # The spectra built from each eigenvector agree far beyond truncation error.
    xi_eigsy = [Q[i, idx] for i in range(A.rows)]
    even = [(xi_eigsy[N + n] + xi_eigsy[N - n]) / 2 for n in range(-N, N + 1)]
    spec_inv = ccm.operator_eigenvalues(mode.eigenvector, N, L, 3)
    spec_eig = ccm.operator_eigenvalues(even, N, L, 3)
    for a, b in zip(spec_inv, spec_eig):
        assert abs(a - b) < 1e-20


def test_forward_spectrum_matches_ordinates():
    # The crux: the prime-built operator's spectrum reproduces the low zeros, with
    # the zeros used only to *check* the output (forward, not inverse). Even at
    # this small N the agreement is already many digits, and the error grows with
    # the index k exactly as the source reports.
    res = ccm.converge(20, mp.sqrt(13), 3, dps=80)
    assert len(res.eigenvalues) == 3
    assert all(e > 0 for e in res.errors)
    assert all(err < mp.mpf(10) ** (-8) for err in res.errors)
    # The first eigenvalue is t_1 to far better than fp64 could represent.
    assert res.errors[0] < mp.mpf(10) ** (-20)
    # Error grows with the zero index (source §6 pattern).
    assert res.errors[0] < res.errors[-1]
    # Weil positivity: the minimal eigenvalue is tiny and positive. This also
    # guards the W_{0,2} - W_R - sum_p W_p combination sign (the all-+ blunder
    # flips eps_N negative and plateaus 5 orders short of the source table).
    assert res.eigenvalue > 0
    assert res.eigenvalue < mp.mpf(10) ** (-30)


def test_convergence_figure_renders(tmp_path):
    # errors_as_float keeps positive magnitudes (the figure applies the log axis);
    # a log10 pre-transform here would feed non-positive data to semilogy.
    mp.mp.dps = 30
    errors = [mp.mpf(10) ** (-k) for k in range(55, 5, -1)]  # 1e-55 .. 1e-6
    floats = ccm.errors_as_float(errors)
    assert all(v > 0 for v in floats)
    out = plots.ccm_convergence_figure(
        {"$x=13$": floats}, N=120, out_path=tmp_path / "ccm.png"
    )
    assert out.exists()
