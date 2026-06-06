"""Connes–Consani–Moscovici finite-cutoff operator — CPU multiprecision reference.

This is the flagship phase's correctness anchor (issue #8). It assembles the
finite-cutoff Weil quadratic form ``QW_lambda^N`` of Connes, Consani & Moscovici
(*Zeta Spectral Triples*, arXiv:2511.22755), extracts the rank-one perturbed
scaling operator ``D_log^{(lambda,N)}`` from its minimal eigenvector, and reads
off that operator's spectrum — the low-lying zeta-zero approximations. Every
formula here implements ``knowledge/ccm-operator.md`` (verified equation by
equation against the source PDF); equation numbers ``(n.m)`` below refer to it.

**Forward, not inverse.** The matrix is built from the *primes* — a von Mangoldt
sum cut at ``k <= lambda^2`` — and from archimedean / point-mass distributions.
The zeta zeros enter *only* at the very end, as the values the computed spectrum
is compared against (:func:`converge`). Nothing here consumes a zero as input;
see ``README.md`` / ``knowledge/project-framing.md``.

**Precision, not throughput.** The reported per-zero agreements reach
``~1e-55..1e-60`` — far below fp64's machine epsilon (``~1e-16``) — so fp64
cannot even *represent* the result. The whole pipeline (fill *and* eigensolve)
runs in ``mpmath`` at ~200 digits; this module is deliberately CPU-only. The
GPU's role (dense special-function fill, many-``lambda`` sweep) is issue #9. See
the precision note in ``CLAUDE.md``.

Headline configuration (source §6): ``N = 120`` (matrix dimension ``2N+1 = 241``),
``lambda in {sqrt(12), sqrt(13), sqrt(14)}``, ``dps ~ 210``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

# ----------------------------------------------------------------------------
# Arithmetic input: prime powers and the von Mangoldt weight
# ----------------------------------------------------------------------------


def prime_powers(x) -> list[tuple[int, mp.mpf]]:
    """Prime powers ``1 < k <= x`` with their von Mangoldt weight ``Lambda(k)``.

    ``Lambda(k) = log p`` when ``k = p^m`` for a prime ``p``, else ``0`` (those
    ``k`` are simply omitted). This is the entire arithmetic content of the
    operator — the cutoff ``x = lambda^2`` is where the primes enter (§4.3). For
    ``x = 13`` this returns ``k in {2, 3, 4, 5, 7, 8, 9, 11, 13}``.
    """
    out: list[tuple[int, mp.mpf]] = []
    kmax = int(mp.floor(x))
    for k in range(2, kmax + 1):
        p = _prime_base(k)
        if p is not None:
            out.append((k, mp.log(p)))
    return out


def _prime_base(k: int) -> int | None:
    """Return ``p`` if ``k = p^m`` for a prime ``p`` and ``m >= 1``, else None."""
    n = k
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            return p if n == 1 else None
        p += 1
    return k  # k is prime (and > 1)


# ----------------------------------------------------------------------------
# The assembly primitive q(U_n, U_m) (Lemma 2.3, eqs. 2.7-2.10)
# ----------------------------------------------------------------------------


def q_kernel(m: int, n: int, y, L) -> mp.mpf:
    """Even quadratic-form kernel ``q(U_m, U_n)(y)`` for ``y in [0, L]`` (§3).

    ``q(U_m, U_n)(y) = (sin(2 pi m y / L) - sin(2 pi n y / L)) / (pi (n - m))``
    for ``m != n``, and ``q(U_n, U_n)(y) = 2 (1 - y / L) cos(2 pi n y / L)``.
    Symmetric in ``(m, n)``; ``q = 0`` at ``y = 0`` off the diagonal, ``= 2`` on
    it. Extended evenly to ``[-L, L]`` (not needed here: all callers pass
    ``y >= 0``).
    """
    L = mp.mpf(L)
    y = mp.mpf(y)
    w = 2 * mp.pi * y / L
    if m == n:
        return 2 * (1 - y / L) * mp.cos(n * w)
    return (mp.sin(m * w) - mp.sin(n * w)) / (mp.pi * (n - m))


# ----------------------------------------------------------------------------
# Archimedean primitives J, K, M (Prop. 4.2, eqs. 4.5-4.7)
# ----------------------------------------------------------------------------
#
# Each is an exact closed form whose only series parameter is q = e^{-2L} < 1,
# so mpmath's hyp2f1 / digamma / polygamma / lerchphi converge geometrically.
# We use these (not direct quadrature) for the production fill; a quadrature
# cross-check lives in archimedean_entry_quad().


def _zk(k: int, L) -> mp.mpc:
    """The shared hypergeometric/digamma argument ``i pi k / L + 1/4``."""
    return mp.mpc(0, mp.pi * k / mp.mpf(L)) + mp.mpf(1) / 4


def _lerch(z, s, a):
    """Hurwitz-Lerch ``Phi(z, s, a) = sum_{k>=0} z^k / (a + k)^s`` by direct sum.

    The only Lerch argument we ever need has ``z = e^{-2L} << 1`` (the source's
    series parameter), so the sum converges geometrically and a plain term-by-term
    accumulation to the working epsilon is both faster and more transparent than
    ``mp.lerchphi``'s general algorithm — exactly as the spec suggests.
    """
    z = mp.mpf(z)
    eps = mp.mpf(10) ** (-(mp.mp.dps + 10))
    total = a * 0  # mpf or mpc depending on a
    z_pow = mp.mpf(1)
    k = 0
    while True:
        term = z_pow / (a + k) ** s
        total += term
        if k > 2 and abs(term) < eps * (abs(total) + 1):
            break
        z_pow *= z
        k += 1
    return total


def archimedean_J(k: int, L) -> mp.mpf:
    """``J(k) = \\int_0^L sin(2 pi k x / L) rho(x) dx`` (eq. 4.5).

    With ``rho(x) = e^{x/2} / (e^x - e^{-x})``. ``J`` is odd in ``k``
    (``J(0) = 0``); this is the building block for every off-diagonal
    archimedean entry.
    """
    L = mp.mpf(L)
    qq = mp.exp(-2 * L)
    zk = _zk(k, L)
    coef = 2 * L / (L + 4 * mp.pi * 1j * k)
    fterm = mp.exp(-L / 2) * mp.im(coef * mp.hyp2f1(1, zk, zk + 1, qq))
    return fterm + mp.im(mp.digamma(zk)) / 2


def archimedean_K(n: int, L) -> mp.mpf:
    """``K(n) = \\int_0^L x cos(2 pi n x / L) rho(x) dx`` (eq. 4.6).

    Even in ``n``. Carries the trigamma term (the correction noted in §9 of the
    spec); enters the diagonal archimedean entry.
    """
    L = mp.mpf(L)
    qq = mp.exp(-2 * L)
    zk = _zk(n, L)
    hyp = mp.hyp2f1(1, zk, zk + 1, qq)
    t1 = -L * mp.exp(-L / 2) * mp.im((2 * L / (4 * mp.pi * n - 1j * L)) * hyp)
    t2 = -(mp.exp(-L / 2) / 4) * mp.re(_lerch(qq, 2, zk))
    t3 = mp.re(mp.polygamma(1, zk)) / 4
    return t1 + t2 + t3


def archimedean_M(n: int, L) -> mp.mpf:
    """``M(n) = \\int_0^L (cos(2 pi n x / L) - 1) rho(x) dx`` (eq. 4.7).

    Even in ``n`` (``M(0) = 0``); enters the diagonal archimedean entry.
    """
    L = mp.mpf(L)
    qq = mp.exp(-2 * L)
    zk = _zk(n, L)
    hyp = mp.hyp2f1(1, zk, zk + 1, qq)
    t1 = -mp.exp(-L / 2) * mp.re((2 * L / (L + 4 * mp.pi * 1j * n)) * hyp)
    t2 = 2 * mp.exp(-L / 2) * mp.hyp2f1(mp.mpf(1) / 4, 1, mp.mpf(5) / 4, qq)
    t3 = -mp.re(mp.digamma(zk) - mp.digamma(mp.mpf(1) / 4)) / 2
    return t1 + t2 + t3


def _archimedean_jkm(n: int, L, *, qq, exp_mhalf, m_const, dig_quarter):
    """Fused ``(J, K, M)`` for mode ``n`` (eqs. 4.5-4.7), sharing the expensive
    pieces the standalone functions each recompute.

    ``archimedean_J``/``_K``/``_M`` independently evaluate the *same*
    ``hyp2f1(1, zk, zk+1, qq)`` (3x) and ``digamma(zk)`` (2x: J and M); across the
    O(N) assembly loop that is the bulk of the archimedean special-function work.
    This computes ``zk``, that one hypergeometric, ``digamma(zk)``, the trigamma and
    the order-2 Lerch once each. The cheap complex coefficient arithmetic is left
    identical to the standalone formulas, so the result is bit-for-bit the same
    (asserted in the tests). The ``n``-independent ``qq = e^{-2L}``,
    ``exp_mhalf = e^{-L/2}``, the M constant ``2 e^{-L/2} 2F1(1/4,1,5/4,qq)`` and
    ``digamma(1/4)`` are hoisted by the caller out of the loop.
    """
    zk = _zk(n, L)
    hyp = mp.hyp2f1(1, zk, zk + 1, qq)
    dig = mp.digamma(zk)
    j = exp_mhalf * mp.im((2 * L / (L + 4 * mp.pi * 1j * n)) * hyp) + mp.im(dig) / 2
    k = (
        -L * exp_mhalf * mp.im((2 * L / (4 * mp.pi * n - 1j * L)) * hyp)
        - (exp_mhalf / 4) * mp.re(_lerch(qq, 2, zk))
        + mp.re(mp.polygamma(1, zk)) / 4
    )
    m = (
        -exp_mhalf * mp.re((2 * L / (L + 4 * mp.pi * 1j * n)) * hyp)
        + m_const
        - mp.re(dig - dig_quarter) / 2
    )
    return j, k, m


def _archimedean_diag_const(L) -> mp.mpf:
    """The ``n``-independent part of the diagonal archimedean entry.

    Equals ``G + 2 C0`` where ``G = gamma + log(4 pi (e^L - 1)/(e^L + 1))`` is the
    boundary term of (4.4) and ``C0 = \\int_0^L (e^{x/2} - 1)/(e^x - e^{-x}) dx``.
    Expanding ``rho(x) = sum_{j>=0} e^{-(2j+1/2) x}`` gives the closed form
    ``C0 = (ln 2)/2 + pi/4 - (e^{-L/2}/2) Phi(e^{-2L},1,1/4)
            + (e^{-L}/2) Phi(e^{-2L},1,1/2)``.

    This term adds a constant ``c`` to every diagonal entry, i.e. ``c * I`` to the
    matrix: it shifts all eigenvalues by ``c`` but leaves the eigenvectors
    untouched, so it does not affect the extracted ``xi`` or the spectrum. It is
    included for fidelity to the paper's matrix (and so the minimal eigenvalue is
    the genuine ``epsilon_N`` rather than a shifted impostor).
    """
    L = mp.mpf(L)
    qq = mp.exp(-2 * L)
    g = mp.euler + mp.log(4 * mp.pi * (mp.exp(L) - 1) / (mp.exp(L) + 1))
    c0 = (
        mp.log(2) / 2
        + mp.pi / 4
        - mp.exp(-L / 2) / 2 * _lerch(qq, 1, mp.mpf(1) / 4)
        + mp.exp(-L) / 2 * _lerch(qq, 1, mp.mpf(1) / 2)
    )
    return g + 2 * c0


def archimedean_entry(n: int, m: int, L, *, j_cache=None) -> mp.mpf:
    """Archimedean matrix entry ``W_R(V_n, V_m)`` (Prop. 4.2).

    Off the diagonal ``(n != m)`` this is the divided difference
    ``(J(m) - J(n)) / (pi (n - m))``; on it, the boundary constant plus
    ``2 M(n) - (2/L) K(n)``. ``j_cache`` (a dict ``k -> J(k)``) lets the
    assembler reuse the ``2N+1`` distinct ``J`` values across the whole matrix.
    """
    if n == m:
        return (
            _archimedean_diag_const(L)
            + 2 * archimedean_M(abs(n), L)
            - 2 / mp.mpf(L) * archimedean_K(abs(n), L)
        )
    jn = _cached_J(n, L, j_cache)
    jm = _cached_J(m, L, j_cache)
    return (jm - jn) / (mp.pi * (n - m))


def _cached_J(k: int, L, j_cache) -> mp.mpf:
    if j_cache is None:
        return archimedean_J(k, L)
    if k not in j_cache:
        j_cache[k] = archimedean_J(k, L)
    return j_cache[k]


def archimedean_entry_quad(n: int, m: int, L, *, extra: int = 40) -> mp.mpf:
    """``W_R(V_n, V_m)`` by direct quadrature of the master formula (4.4).

    The robust cross-check the spec recommends for the delicate diagonal: it
    evaluates ``(omega(0)/2)(gamma + log(4 pi (e^L-1)/(e^L+1)))
    + \\int_0^L [e^{x/2} omega(x) - omega(0)] / (e^x - e^{-x}) dx`` with
    ``omega = q(U_n, U_m)``, raising the working precision by ``extra`` digits so
    the removable ``x -> 0`` singularity survives the tanh-sinh nodes. Slower and
    less accurate than the closed form, used only in tests.
    """
    L = mp.mpf(L)
    with mp.workdps(mp.mp.dps + extra):
        w0 = q_kernel(m, n, 0, L)
        boundary = (w0 / 2) * (
            mp.euler + mp.log(4 * mp.pi * (mp.exp(L) - 1) / (mp.exp(L) + 1))
        )

        def integrand(x):
            x = mp.mpf(x)
            num = mp.exp(x / 2) * q_kernel(m, n, x, L) - w0
            return num / (mp.exp(x) - mp.exp(-x))

        integral = mp.quad(integrand, [0, L])
    return +(boundary + integral)


# ----------------------------------------------------------------------------
# Point-mass entry W_{0,2} (Lemma 4.1, eq. 4.2)
# ----------------------------------------------------------------------------


def point_mass_entry(n: int, m: int, L) -> mp.mpf:
    """Point-mass contribution ``W_{0,2}(V_n, V_m)`` (eq. 4.2, closed form)."""
    L = mp.mpf(L)
    pref = 32 * L * mp.sinh(L / 4) ** 2
    num = L**2 - 16 * mp.pi**2 * n * m
    den = (L**2 + 16 * mp.pi**2 * m**2) * (L**2 + 16 * mp.pi**2 * n**2)
    return pref * num / den


# ----------------------------------------------------------------------------
# Full Weil matrix assembly
# ----------------------------------------------------------------------------


def assemble_weil_matrix(N: int, lam) -> mp.matrix:
    """Assemble the real-symmetric ``QW_lambda^N`` (dimension ``2N+1``) at the
    current ``mpmath`` precision.

    Row/column index ``i in {0, ..., 2N}`` is the scaling mode ``n = i - N`` (so
    ``i = N`` is ``n = 0``). The entry follows the Weil functional's sign pattern
    (eqs. 3.10/3.13 of the source): ``QW = W_{0,2} - W_R - sum_p W_p`` — the
    point-mass (4.2) *minus* the archimedean (4.4-4.7) *minus* the prime sum (4.3).
    ``lam`` is ``lambda`` (e.g. ``mp.sqrt(13)``); the prime cutoff is
    ``x = lambda^2`` and ``L = 2 log lambda``.

    Both the archimedean *and* the prime off-diagonal entries are divided
    differences of a single odd function of the mode index — the Lemma 5.1
    structure ``tau_{i,j} = (b_i - b_j)/(i - j)``. We exploit that: precompute one
    array ``B[n] = J(n) + sum_k coef_k sin(2 pi n log k / L)`` and fill every
    off-diagonal arch+prime entry as ``(B[m] - B[n]) / (pi (n - m))``, turning the
    naive ``O(N^2 * #primes)`` fill into a cheap ``O(N^2)`` one.
    """
    lam = mp.mpf(lam)
    L = 2 * mp.log(lam)
    dim = 2 * N + 1

    pps = prime_powers(lam**2)
    logs = [mp.log(k) for k, _ in pps]
    coefs = [lam_k / mp.sqrt(k) for (k, lam_k) in pps]

    # B[n] (odd) collects J(n) and the prime sin-sum; the diagonal collects the
    # archimedean diagonal entry and the prime cos-sum (both even in n). The three
    # archimedean integrals are fused (one hyp2f1 / digamma per mode instead of
    # 3 / 2) and their n-independent factors hoisted out of the loop (#17).
    b_pos: list[mp.mpf] = []
    diag_const = _archimedean_diag_const(L)
    qq = mp.exp(-2 * L)
    exp_mhalf = mp.exp(-L / 2)
    m_const = 2 * exp_mhalf * mp.hyp2f1(mp.mpf(1) / 4, 1, mp.mpf(5) / 4, qq)
    dig_quarter = mp.digamma(mp.mpf(1) / 4)
    diag_total: list[mp.mpf] = []
    for n in range(N + 1):
        sin_sum = mp.mpf(0)
        cos_sum = mp.mpf(0)
        for yk, coef in zip(logs, coefs):
            wk = 2 * mp.pi * n * yk / L
            sin_sum += coef * mp.sin(wk)
            cos_sum += coef * 2 * (1 - yk / L) * mp.cos(wk)
        j, k, m = _archimedean_jkm(
            n, L, qq=qq, exp_mhalf=exp_mhalf, m_const=m_const, dig_quarter=dig_quarter
        )
        b_pos.append(j + sin_sum)
        diag_total.append(diag_const + 2 * m - 2 / L * k + cos_sum)
    # Mirror to all modes n = -N..N at index n + N (B is odd).
    B = [(-b_pos[-(i - N)] if i - N < 0 else b_pos[i - N]) for i in range(dim)]

    # Point-mass precompute: W_{0,2}(n,m) = pref (L^2 - 16 pi^2 n m) a_n a_m.
    pref = 32 * L * mp.sinh(L / 4) ** 2
    L2 = L**2
    fourpi2 = 16 * mp.pi**2
    a = [1 / (L2 + fourpi2 * (i - N) ** 2) for i in range(dim)]

    # QW = W_{0,2} - W_R - sum_p W_p: point-mass minus the (archimedean + prime)
    # contribution carried by B / diag_total (eqs. 3.10, 3.13).
    A = mp.zeros(dim, dim)
    for i in range(dim):
        n = i - N
        an = a[i]
        Bn = B[i]
        for j in range(i, dim):
            m = j - N
            pm = pref * (L2 - fourpi2 * n * m) * an * a[j]
            if i == j:
                val = pm - diag_total[abs(n)]
            else:
                val = pm - (B[j] - Bn) / (mp.pi * (n - m))
            A[i, j] = val
            A[j, i] = val
    return A


# ----------------------------------------------------------------------------
# Minimal eigenvector via inverse iteration
# ----------------------------------------------------------------------------


@dataclass
class MinimalMode:
    """The minimal eigenpair of ``QW_lambda^N`` and its parity diagnostics."""

    eigenvalue: mp.mpf  # epsilon_N (Rayleigh): tiny and >= 0 (Weil positivity)
    eigenvector: list  # xi components, index i -> mode n = i - N, length 2N+1
    parity_residual: mp.mpf  # max |xi_{-n} - xi_n| / max|xi|; ~0 confirms even


def _even_block(A: mp.matrix, N: int) -> mp.matrix:
    """Restriction of the bisymmetric ``QW`` to its even (parity ``+1``) subspace.

    ``QW`` commutes with the parity grading ``gamma`` (``A[i, j] = A[2N-i, 2N-j]``,
    asserted in ``test_weil_matrix_symmetric_and_parity_even``), so in the
    orthonormal even basis ``{e_0 = delta_0, u_n = (delta_n + delta_{-n})/sqrt 2}``
    (``n = 1..N``) it block-diagonalises to an ``(N+1) x (N+1)`` matrix carrying the
    whole *even* spectrum — including the minimal eigenvalue ``epsilon_N``, whose
    eigenvector is even (Def. 5.3). The block entries are

        B[0, 0]   = A[N, N]                       (the n = 0 mode)
        B[0, k]   = sqrt 2 * A[N, N+k]            (k = 1..N)
        B[j, k]   = A[N+j, N+k] + A[N+j, N-k]     (j, k = 1..N).

    Solving the near-null mode in this block is the same eigenproblem at half the
    dimension, so the ``O(N^3)`` LU shrinks by ``((N+1)/(2N+1))^3 ~ 1/8`` (#18).
    """
    c = N  # matrix index of mode n = 0
    sqrt2 = mp.sqrt(2)
    block = mp.zeros(N + 1, N + 1)
    block[0, 0] = A[c, c]
    for k in range(1, N + 1):
        val = sqrt2 * A[c, c + k]
        block[0, k] = val
        block[k, 0] = val
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            val = A[c + j, c + k] + A[c + j, c - k]
            block[j, k] = val
            block[k, j] = val
    return block


def _reconstruct_even(coeffs, N: int) -> list:
    """Lift a reduced even-block vector ``coeffs`` (length ``N+1``) to the full
    ``2N+1`` mode vector, indexed ``i -> n = i - N``.

    The orthonormal even basis maps ``coeffs[0] -> xi_0`` and
    ``coeffs[k] -> xi_{+/-k} = coeffs[k]/sqrt 2`` (``k = 1..N``); the ``sqrt 2``
    keeps the lift norm-preserving, so a unit ``coeffs`` yields a unit ``xi``.
    """
    sqrt2 = mp.sqrt(2)
    full = [mp.mpf(0)] * (2 * N + 1)
    full[N] = coeffs[0]
    for k in range(1, N + 1):
        v = coeffs[k] / sqrt2
        full[N + k] = v
        full[N - k] = v
    return full


def smallest_even_eigenvector(A: mp.matrix, N: int, *, iters: int = 6) -> MinimalMode:
    """Minimal eigenvector ``xi`` of ``QW_lambda^N`` by inverse iteration.

    ``QW`` is positive definite with a single tiny eigenvalue ``epsilon_N`` (the
    near-null mode whose eigenvector defines the operator) and the rest ``O(1)``;
    inverse iteration (``x <- A^{-1} x``) converges to that near-null direction in
    a single step because the spectral gap ``epsilon_2 / epsilon_N`` is enormous.

    The minimal eigenvector is even (``gamma xi = xi``, Def. 5.3), so we iterate in
    the ``(N+1)``-dimensional even block (:func:`_even_block`) rather than the full
    ``2N+1`` space: the ``O(N^3)`` LU factorisation — ~81% of the pipeline after the
    #17 factor-once win — runs on half the dimension, ~8x fewer flops (#18). The
    even basis makes the result exactly even by construction (no symmetrisation, no
    odd contamination to project out), so the parity residual is identically zero.
    As in #17 the block is factored once (LU) and reused across all ``iters`` solves.
    """
    block = _even_block(A, N)
    lu, piv = mp.mp.LU_decomp(block, overwrite=False)
    c = mp.matrix([mp.mpf(1) for _ in range(N + 1)])
    for _ in range(iters):
        c = mp.mp.U_solve(lu, mp.mp.L_solve(lu, c, piv))
        c = c / mp.norm(c)

    even = _reconstruct_even([c[i] for i in range(N + 1)], N)
    # Rayleigh quotient for epsilon_N against the full QW (cheap O(dim^2)); this
    # keeps epsilon_N defined against the paper's matrix and equals the reduced
    # quotient c^T block c.
    dim = 2 * N + 1
    xv = mp.matrix(even)
    Ax = A * xv
    eps = mp.fsum(even[i] * Ax[i] for i in range(dim)) / mp.fsum(
        even[i] ** 2 for i in range(dim)
    )
    return MinimalMode(eigenvalue=eps, eigenvector=even, parity_residual=mp.mpf(0))


def _smallest_even_eigenvector_full(
    A: mp.matrix, N: int, *, iters: int = 6
) -> MinimalMode:
    """Full-space inverse iteration (the pre-#18 path), retained as the cross-check.

    Identical algorithm to :func:`smallest_even_eigenvector` but in the full
    ``2N+1`` space: factor ``A`` once, inverse-iterate, *then* symmetrise to the even
    subspace and report the pre-symmetrisation parity residual. The #18 reduced path
    must agree with this to the precision floor (asserted in the tests); it is also
    what the GPU-conditioning study (#9) calls.
    """
    dim = 2 * N + 1
    lu, piv = mp.mp.LU_decomp(A, overwrite=False)
    x = mp.matrix([mp.mpf(1) for _ in range(dim)])
    for _ in range(iters):
        x = mp.mp.U_solve(lu, mp.mp.L_solve(lu, x, piv))
        x = x / mp.norm(x)

    Ax = A * x
    eps = mp.fdot(x, Ax) / mp.fdot(x, x)

    vec = [x[i] for i in range(dim)]
    parity = _parity_residual(vec, N)
    even = [(vec[N + n] + vec[N - n]) / 2 for n in range(-N, N + 1)]
    nrm = mp.norm(mp.matrix(even))
    even = [v / nrm for v in even]
    return MinimalMode(eigenvalue=eps, eigenvector=even, parity_residual=parity)


def _parity_residual(vec, N: int) -> mp.mpf:
    scale = max(abs(v) for v in vec)
    worst = max(abs(vec[N + n] - vec[N - n]) for n in range(1, N + 1))
    return worst / scale


# ----------------------------------------------------------------------------
# Spectrum of the rank-one perturbed scaling operator (Theorem 1.1)
# ----------------------------------------------------------------------------


def operator_eigenvalues(xi, N: int, L, count: int) -> list[mp.mpf]:
    """First ``count`` positive eigenvalues of ``D_log^{(lambda,N)}``.

    The rank-one perturbed operator ``D_log - |D_log xi><delta_N|`` (Theorem 1.1)
    has, besides a trivial ``z = 0``, eigenvalues that are the roots of the
    secular function

        F(z) = sum_{n=-N}^{N} xi_n / (d_n - z),   d_n = 2 pi n / L,

    (the determinant condition; ``delta_N`` carries unit mode coefficients, so it
    factors out and the root set is independent of the ``delta_N(xi) = 1``
    normalisation). The real roots are the zeta-zero approximations. ``F`` has a
    pole at each ``d_n``; we subdivide the positive axis at those poles and find
    every root strictly inside each gap.

    The roots do *not* interlace the poles (the zero density is below the uniform
    pole density, so a gap holds 0, 1 or occasionally 2 roots), and a root can sit
    arbitrarily close to a pole. So within each gap we sample on a grid that is
    **clustered geometrically toward both poles** as well as spread across the
    middle — a uniform grid misses the near-pole roots and silently drops one,
    which would misalign every higher eigenvalue. Cross-checked against a direct
    rank-one eigensolve (``mp.eig``) in the tests.

    The sum is folded over parity. With ``xi`` even (``xi[N+n] = xi[N-n]``) and the
    poles ``d_n = 2 pi n / L`` odd in ``n``, pairing ``+n`` with ``-n`` gives

        F(z) = -xi_0 / z + sum_{n=1}^{N} 2 z xi_n / (d_n^2 - z^2),

    half the terms of the raw ``(2N+1)``-term sum and ~2x per evaluation across the
    pole-clustered search — the roots are unchanged (#18).
    """
    L = mp.mpf(L)
    xi0 = xi[N]
    xs = [xi[N + n] for n in range(1, N + 1)]  # xi_n, n = 1..N (xi is even)
    d2 = [(2 * mp.pi * n / L) ** 2 for n in range(1, N + 1)]  # d_n^2

    def F(z):
        zz = z * z
        two_z = 2 * z
        return -xi0 / z + mp.fsum(two_z * xs[i] / (d2[i] - zz) for i in range(N))

    pos_poles = [2 * mp.pi * n / L for n in range(1, N + 1)]
    edges = [mp.mpf(0)] + pos_poles
    fractions = _gap_fractions()
    roots: list[mp.mpf] = []
    for a, b in zip(edges, edges[1:]):
        roots.extend(_roots_in_gap(F, a, b, fractions))
        if len(roots) >= count:
            break
    roots.sort()
    return roots[:count]


def _gap_fractions() -> list[mp.mpf]:
    """Sample fractions in ``(0, 1)`` clustered toward both ends plus a middle grid.

    The geometric clustering (down to ``1e-13`` of the gap from each pole) is what
    captures roots that sit within a hair of a pole; the uniform middle catches the
    ordinary interior roots (and the rare two-per-gap case).
    """
    near = [mp.mpf(10) ** (-k) for k in range(13, 2, -1)]  # 1e-13 .. 1e-3
    middle = [mp.mpf(j) / 24 for j in range(1, 24)]  # 1/24 .. 23/24
    fr = sorted(set(near) | set(middle) | {1 - x for x in near})
    return [f for f in fr if 0 < f < 1]


def _roots_in_gap(F, a, b, fractions) -> list[mp.mpf]:
    """Bracket and refine every sign change of ``F`` strictly inside ``(a, b)``."""
    width = b - a
    pts = [a + width * f for f in fractions]
    vals = [F(p) for p in pts]
    found: list[mp.mpf] = []
    for (p0, f0), (p1, f1) in zip(zip(pts, vals), zip(pts[1:], vals[1:])):
        if f0 == 0:
            found.append(p0)
        elif (f0 > 0) != (f1 > 0):
            found.append(mp.findroot(F, (p0, p1), solver="anderson"))
    return found


# ----------------------------------------------------------------------------
# Top-level: errors against the true zeros
# ----------------------------------------------------------------------------


@dataclass
class CCMResult:
    """Outcome of one finite-cutoff convergence experiment."""

    N: int
    lam: mp.mpf
    eigenvalue: mp.mpf  # epsilon_N of QW_lambda^N
    parity_residual: mp.mpf
    eigenvalues: list[mp.mpf]  # computed positive spectrum (zero approximations)
    zeros: list[mp.mpf]  # true ordinates t_k
    errors: list[mp.mpf]  # |eig_k - t_k|


def reference_ordinates(count: int) -> list[mp.mpf]:
    """First ``count`` ordinates ``t_k`` of ``zeta(1/2 + i t)`` (mpmath, current
    precision). Used only to *check* the computed spectrum — forward."""
    return [mp.im(mp.zetazero(k)) for k in range(1, count + 1)]


def operator_spectrum(
    N: int, lam, *, count: int | None = None, dps: int = 210
) -> list[mp.mpf]:
    """The operator's positive spectrum (the zeta-zero approximations), no zeros.

    Like :func:`converge` but returns only the computed eigenvalues — the forward
    output — without comparing to the true ordinates. ``count`` defaults to ``N``
    (about as many low levels as the dimension ``2N+1`` supports), which is what
    the universality diagnostics (#5/#6/#15) want: a whole spectrum to unfold by
    its own count and feed to the spacing statistics, not just the first few.
    """
    if count is None:
        count = N
    with mp.workdps(dps):
        lam = mp.mpf(lam)
        L = 2 * mp.log(lam)
        A = assemble_weil_matrix(N, lam)
        mode = smallest_even_eigenvector(A, N)
        return operator_eigenvalues(mode.eigenvector, N, L, count)


def converge(N: int, lam, count: int, *, dps: int = 210) -> CCMResult:
    """Run the full forward experiment for one ``(N, lambda)`` at ``dps`` digits.

    Assembles ``QW_lambda^N``, extracts the minimal even eigenvector ``xi``, forms
    the rank-one operator, takes its first ``count`` positive eigenvalues, and
    compares them to the true ordinates. Returns everything needed to reproduce
    the source's §6 convergence table and to plot ``|eig_k - t_k|`` (#8).
    """
    with mp.workdps(dps):
        lam = mp.mpf(lam)
        L = 2 * mp.log(lam)
        A = assemble_weil_matrix(N, lam)
        mode = smallest_even_eigenvector(A, N)
        eigs = operator_eigenvalues(mode.eigenvector, N, L, count)
        zeros = reference_ordinates(count)
        errors = [abs(e - t) for e, t in zip(eigs, zeros)]
    return CCMResult(
        N=N,
        lam=lam,
        eigenvalue=mode.eigenvalue,
        parity_residual=mode.parity_residual,
        eigenvalues=eigs,
        zeros=zeros,
        errors=errors,
    )


def errors_as_float(errors: list[mp.mpf]) -> np.ndarray:
    """Per-zero errors as float64, for plotting on a log axis.

    The magnitudes run down to ~1e-60, comfortably inside float64's range, so the
    values survive the cast (only their ~200-digit relative precision is dropped,
    immaterial for a log-scale figure). Tables keep the full ``mpf`` errors."""
    return np.array([float(e) for e in errors])
