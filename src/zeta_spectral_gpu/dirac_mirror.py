"""Sierra's prime-driven massless-Dirac model — a forward locator (issue #25).

Germán Sierra, *The Riemann zeros as energy levels of a Dirac fermion in a
potential built from the prime numbers in Rindler spacetime*, J. Phys. A **47**,
325204 (2014), arXiv:1404.4252; reviewed in arXiv:1601.01797 §X–XV.

A massless Dirac fermion in the right Rindler wedge (radial coordinate
``rho >= l_1 = 1``) is free except for delta-function "moving mirrors" at radial
positions ``l_n``, each a unitary 2x2 transfer matrix ``T_n`` with reflection
coefficient ``varrho_n`` (eq. 10.20). Amplitudes iterate from the boundary vector
``|A_1(vartheta)> = (1, e^{i vartheta})`` (eq. 10.19/10.22). The **prime-driven**
choice (eq. 11.4 / 13.5) places a mirror at every square-free integer,

    l_n = sqrt(n),     varrho_n = mu(n) / sqrt(n)        (mu = Moebius),

so the mirror for a prime ``p`` sits at ``sqrt(p)`` and the boundary->mirror
round trip has period ``log p`` — Berry's "primes as periodic orbits", made
concrete and energy-independent (the fermion is massless).

Forward, not inverse — and the distinction matters more here than anywhere else
in the repo (see ``_private/issue-25-forward-ruling.md`` and project-framing).
The model has two separable pieces with *opposite* forward status:

* **Piece A — the prime-built locator (FORWARD).** The decisive object is the
  finite, purely number-theoretic partial sum (eq. 12.20)

      M'_z(n) = sum_{k<=n} mu(k) k^{-z},   z = 1/2 + iE,   E real,

  whose magnitude *grows with the truncation n* exactly when ``E`` is a zero of
  zeta (Fig. 14), with the quantitative prediction ``|M'_z(n)| ~ log n / |Z'(E)|``
  at a simple zero (eq. 12.30). This consumes only ``mu(k)`` and ``k^{-z}`` —
  pure number theory — and the zeros come out. Same posture as the CCM flagship.
  ``mobius_partial_sum`` / ``growth_profile`` implement it; it is the forward
  deliverable.

* **Piece B — the per-zero ``vartheta`` tuning (the inverse trap).** Individual
  zeros become genuine normalizable eigenstates of the self-adjoint ``H_vartheta``
  only when the boundary phase is set to ``vartheta = -(theta(E) + (pi/2) sign
  Z'(E))`` (eq. 12.33), with ``theta`` the Riemann–Siegel theta and ``Z`` the
  Hardy function. Both are *explicit* functions of a real ``E``, but feeding a
  **known** zero in to set ``vartheta`` and then "recovering" it consumes the
  zero. ``normalizable_amplitude`` reproduces Sierra's Fig. 4 dichotomy
  (normalizable at a zero with tuned ``vartheta`` vs. continuum otherwise) purely
  to **validate the bound-state machinery** — it is *not* a zero-finder. The
  forward locator is Piece A.

Precision is the opposite regime from the flagship: ``|M'_z(n)|`` is ``O(1)``,
no catastrophic cancellation, so **fp64 is sufficient** (a test cross-checks
against mpmath). That is what makes the GPU version (a Moebius-weighted Dirichlet
partial sum over a dense ``E``-grid, plus the Dirichlet-L generalisation
``mu(n) -> chi(n) mu(n)``) a clean fp64 win, unlike the flagship's mpmath-bound
eigensolve. This module is the CPU reference; the GPU assembly is the follow-up.
"""

from __future__ import annotations

import numpy as np


def mobius_sieve(n_max: int) -> np.ndarray:
    """Möbius function ``mu(1..n_max)`` as an ``int8`` array (index 0 unused/0).

    Linear-ish sieve over smallest prime factors: ``mu(1)=1``; ``mu(n)=0`` if a
    prime square divides ``n``; else ``(-1)^{# distinct prime factors}``. The
    only number-theoretic input the model consumes — everything forward flows
    from this and the ``k^{-z}`` weights.
    """
    if n_max < 1:
        return np.zeros(1, dtype=np.int8)
    mu = np.ones(n_max + 1, dtype=np.int8)
    mu[0] = 0
    is_comp = np.zeros(n_max + 1, dtype=bool)
    primes: list[int] = []
    for i in range(2, n_max + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n_max:
                break
            is_comp[ip] = True
            if i % p == 0:
                mu[ip] = 0  # p^2 | ip
                break
            mu[ip] = -mu[i]
    return mu


def mobius_partial_sum(
    E: np.ndarray | float,
    n: int,
    *,
    sigma: float = 0.5,
    weights: np.ndarray | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``M'_z(n) = sum_{k<=n} c(k) k^{-(sigma+iE)}`` — the forward locator (Piece A).

    Vectorised over ``E`` (fp64 ``complex128``). ``c(k)`` defaults to ``mu(k)``
    (zeta); pass ``weights = chi(k) mu(k)`` for a Dirichlet ``L``-function
    (eq. 13.6 / 14.9 — see :func:`dirichlet.lfunction_weights`). Real ``weights``
    (zeta, principal/quadratic characters) stay real; a genuinely complex
    character gives ``complex128`` weights, handled transparently. The series is
    the truncated Dirichlet series of ``1/zeta(sigma+iE)`` (or ``1/L``, eq. 12.11):
    ``|M'_z(n)|`` grows with ``n`` at a zero on the critical line ``sigma = 1/2``
    and stays bounded / oscillates otherwise.

    No boundary phase ``vartheta`` and no zeros enter — this is the prime-only
    object the zeros are read *off* of.
    """
    E = np.asarray(E, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    if weights is None:
        if mu is None:
            mu = mobius_sieve(n)
        weights = mu[1 : n + 1].astype(np.float64)
    else:
        weights = np.asarray(weights[:n])  # preserve real/complex dtype (L-functions)
    # k^{-(sigma+iE)} = k^{-sigma} * exp(-iE ln k); outer over (E, k).
    amp = weights * k**-sigma  # shape (n,)
    logk = np.log(k)  # shape (n,)
    phase = np.exp(-1j * np.multiply.outer(E, logk))  # (E.shape, n)
    return phase @ amp.astype(np.complex128)


def growth_profile(
    E: float,
    truncations: np.ndarray,
    *,
    sigma: float = 0.5,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``|M'_z(n)|`` at each truncation ``n`` in ``truncations`` (Fig. 14).

    Computes the cumulative partial sum once and samples it, so passing many
    truncations is cheap. The forward signature of a zero is a profile that
    *increases* with ``n``; off a zero it stays bounded.
    """
    truncations = np.asarray(truncations, dtype=np.int64)
    n_top = int(truncations.max())
    if mu is None:
        mu = mobius_sieve(n_top)
    k = np.arange(1, n_top + 1, dtype=np.float64)
    terms = (
        mu[1 : n_top + 1].astype(np.float64) * k**-sigma * np.exp(-1j * E * np.log(k))
    )
    cumulative = np.cumsum(terms)
    return np.abs(cumulative[truncations - 1])


def normalizable_amplitude(
    E: float,
    truncations: np.ndarray,
    *,
    eps: float = 0.25,
    vartheta: float | None = None,
    sigma: float = 0.5,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """``<A_n|A_n>`` at each truncation ``n`` (Sierra's Fig. 4) — the bound-state bridge.

    The leading Magnus / ``eps -> 0`` closed form (eq. 12.12/12.14): writing
    ``M = M'_z(n)`` (``mobius_partial_sum``) and its phase ``Phi = -arg M``,

        <A_n|A_n> = 2 [ e^{-2 eps|M|} cos^2(d) + e^{+2 eps|M|} sin^2(d) ],
        d = (vartheta - Phi) / 2.

    This bridges Piece A (``|M'_z|`` growth) to the *physical* statement — a
    normalizable state. At a zero with ``vartheta`` tuned to the limiting phase
    (``riemann_tuning_phase(E)``, the default) the growing ``sin^2`` mode is
    killed and the norm density *decays* like ``e^{-2 eps|M|} = n^{-2 eps/|Z'(E)|}``
    (sum -> ``2 zeta(1 + 2 eps/|Z'(E)|)`` < inf, eq. 12.34); at a generic ``E`` the
    ``sin^2`` term survives and it stays ``O(1)`` (continuum).

    **Validation utility, not a forward zero-finder.** The default ``vartheta``
    uses ``theta(E)`` — Piece B of the forward ruling — to confirm the machinery
    reproduces the paper. Forward zero-location is Piece A
    (``mobius_partial_sum`` / ``growth_profile``). The full finite-``eps``
    transfer-matrix product (resonances) is a separate, later step.
    """
    if vartheta is None:
        vartheta = riemann_tuning_phase(E)
    truncations = np.asarray(truncations, dtype=np.int64)
    n_top = int(truncations.max())
    if mu is None:
        mu = mobius_sieve(n_top)
    k = np.arange(1, n_top + 1, dtype=np.float64)
    terms = (
        mu[1 : n_top + 1].astype(np.float64) * k**-sigma * np.exp(-1j * E * np.log(k))
    )
    M = np.cumsum(terms)[truncations - 1]
    absM = np.abs(M)
    d = 0.5 * (vartheta - (-np.angle(M)))
    return 2.0 * (
        np.exp(-2.0 * eps * absM) * np.cos(d) ** 2
        + np.exp(2.0 * eps * absM) * np.sin(d) ** 2
    )


# --- RH by contradiction (Sierra sec. XII C): a zero off the line cannot exist --


def offline_mobius_sum(
    E: float,
    truncations: np.ndarray,
    *,
    sigma_c: float,
    e_c: float,
) -> np.ndarray:
    """``M_z(x)`` for a *hypothetical* zero ``rho_c = sigma_c + i e_c`` off the line.

    Sierra's RH-by-contradiction argument (sec. XII C, eq. 12.35): if zeta had a
    zero at ``rho_c`` with ``sigma_c > 1/2``, the truncated Moebius sum would be
    dominated by the residues at ``rho_c`` and its conjugate,

        M_z(x) -> x^{rho_c - z} / ((rho_c - z) zeta'(rho_c))
                + x^{conj(rho_c) - z} / ((conj(rho_c) - z) zeta'(conj(rho_c))),

    with ``z = 1/2 + iE``. Since ``Re(rho_c - z) = sigma_c - 1/2 > 0``, ``|M_z(x)|``
    grows *polynomially* like ``x^{sigma_c - 1/2}`` -- in contrast to the ``log x``
    growth at an on-line zero (eq. 12.30). That fast growth makes the eigenstate
    non-normalizable for *every* self-adjoint extension ``vartheta``
    (:func:`norm_partial_sums`), contradicting the self-adjointness of
    ``H_vartheta`` -- the heuristic "proof" of the Riemann hypothesis.

    Forward, not inverse: ``rho_c`` is a *counterfactual* we plant to probe the
    structure, and ``zeta'(rho_c)`` is just an explicit-function evaluation at that
    hypothetical point -- no true zero of zeta is consumed.
    """
    import mpmath as mp

    z = complex(0.5, E)
    rho = complex(sigma_c, e_c)
    zp = complex(mp.zeta(mp.mpc(sigma_c, e_c), derivative=1))
    x = np.asarray(truncations, dtype=np.float64)
    term = x ** (rho - z) / ((rho - z) * zp)
    term_conj = x ** (rho.conjugate() - z) / ((rho.conjugate() - z) * zp.conjugate())
    return term + term_conj


def norm_partial_sums(
    mobius_values: np.ndarray,
    *,
    eps: float = 0.25,
    vartheta: float = 0.0,
) -> np.ndarray:
    """Cumulative norm estimate ``N_z(N)`` (Sierra eq. 12.14) from ``M_z(1..N)``.

        N_z(N) = sum_{n=1}^{N} (1/n) [ e^{-2 eps |M_n|} (1 + cos(vartheta - Phi_n))
                                     + e^{+2 eps |M_n|} (1 - cos(vartheta - Phi_n)) ],

    with ``M_n = mobius_values[n-1]`` and ``Phi_n = -arg(M_n)`` (eq. 12.13). The
    second (``e^{+2 eps|M|}``) mode is killed only if ``|M_n|`` diverges fast enough
    *and* ``Phi_n -> vartheta`` (eq. 12.19): the on-line zero with ``vartheta`` tuned
    to eq. 12.33 gives a finite norm (eq. 12.34). For an off-line zero
    (:func:`offline_mobius_sum`) ``|M_n| ~ n^{sigma_c-1/2}`` grows while ``Phi_n``
    keeps oscillating, so no ``vartheta`` can cancel the growing mode and ``N_z``
    diverges -- the contradiction with self-adjointness. Returns the running
    cumulative sum so (non)convergence is visible.
    """
    m = np.asarray(mobius_values)
    abs_m = np.abs(m)
    phi = -np.angle(m)
    n = np.arange(1, m.size + 1, dtype=np.float64)
    c = np.cos(vartheta - phi)
    terms = (
        np.exp(-2.0 * eps * abs_m) * (1.0 + c) + np.exp(2.0 * eps * abs_m) * (1.0 - c)
    ) / n
    return np.cumsum(terms)


# --- Explicit functions used for VALIDATION only (Piece B; never zero-lookup) ---
#
# theta(E) (Riemann–Siegel) and sign Z'(E) (Hardy) are elementary functions of a
# real argument, computable with no table of zeros. They appear here to (a) tune
# vartheta when reproducing Fig. 4 and (b) check the |M'_z(n)| ~ log n / |Z'(E)|
# growth-rate prediction (eq. 12.30). They must never enter a pipeline whose
# purpose is to produce zeros — that is the inverse trap the ruling forbids.


def riemann_siegel_theta(E: float) -> float:
    """Riemann–Siegel theta ``theta(E)`` via mpmath ``siegeltheta`` (``mpf``)."""
    import mpmath as mp

    return float(mp.siegeltheta(E))


def hardy_z_prime(E: float, *, h: float = 1e-3) -> float:
    """``Z'(E)`` of the Hardy function ``Z`` (central difference, mpmath ``siegelz``)."""
    import mpmath as mp

    return float((mp.siegelz(E + h) - mp.siegelz(E - h)) / (2 * h))


def riemann_tuning_phase(E: float) -> float:
    """``vartheta(E) = -(theta(E) + (pi/2) sign Z'(E))`` (eq. 12.33) — Piece B.

    The self-adjoint-extension phase that makes a zero ``E`` a normalizable state.
    Validation-only (see ``transfer_amplitudes``); not part of forward
    zero-finding.
    """
    return -(riemann_siegel_theta(E) + (np.pi / 2.0) * np.sign(hardy_z_prime(E)))


# ---------------------------------------------------------------------------
# Exact finite-eps transfer-matrix machinery (#44 spike, gated by #45)
#
# Everything above lives in the eps -> 0 *semiclassical* limit: the
# Baker-Campbell-Hausdorff / Trotter collapse of the transfer-matrix product to
# the Moebius-Dirichlet locator (Sierra eqs. 121-128). The functions below form
# the **exact** product T_k...T_2 at finite eps (eq. 115) -- the un-linearised
# object #44 asks about -- and the harmonic model (Appendix A), the one model
# with a genuine finite-eps spectral density, as a no-prime positive control.
# The spike uses these to rule on whether an operator-intrinsic, non-circular
# finite-eps statistic exists; see ``_private/issue-44-resonance-ruling.md``.
# ---------------------------------------------------------------------------


def transfer_matrix(E: complex, varrho: complex, ell: float) -> np.ndarray:
    """Exact single-mirror transfer matrix ``T(E, varrho, ell)`` (Sierra eq. 115).

    The SU(1,1) matrix propagating the Dirac spinor amplitude across the mirror
    at radial position ``ell`` with (generally complex) reflection amplitude
    ``varrho`` (``|varrho| < 1``)::

        T = 1/(1-|p|^2) [[1+|p|^2,        2 p ell^{-2iE} ],
                         [2 conj(p) ell^{2iE},  1+|p|^2  ]]   (p = varrho).

    ``det T = 1`` and ``T`` is pseudo-unitary w.r.t. ``sigma_z``
    (``T^dag sigma_z T = sigma_z``); the spike's tests check both invariants.
    ``E`` may be complex (the off-diagonal entries are the analytic continuation,
    not conjugates), so the resolvent / resonance question can be probed off the
    real axis.
    """
    denom = 1.0 - abs(varrho) ** 2
    diag = (1.0 + abs(varrho) ** 2) / denom
    off01 = 2.0 * varrho * ell ** (-2j * E) / denom
    off10 = 2.0 * np.conj(varrho) * ell ** (2j * E) / denom
    return np.array([[diag, off01], [off10, diag]], dtype=np.complex128)


def amplitude_norms(
    E: np.ndarray | float,
    varrho: np.ndarray,
    ell: np.ndarray,
    *,
    vartheta: float = 0.0,
) -> np.ndarray:
    """Exact per-interval norms ``<A_k|A_k>``, k = 1..N (Sierra eq. 118 summand).

    Forms the **exact** product ``|A_k> = T_k^{-1} ... T_2^{-1} |A_1(vartheta)>``
    with ``|A_1(vartheta)> = (1, e^{i vartheta})`` (eqs. 117, 120), iterating the
    single-mirror matrices ``T_n = transfer_matrix(E, varrho[n], ell[n])`` over
    mirrors ``n = 2..N``. ``varrho`` and ``ell`` are 1-indexed arrays of length
    ``N + 1`` (index 0 unused; index 1 is the boundary ``ell_1 = 1`` and carries
    no mirror). A mirror with ``varrho[n] == 0`` (a non-square-free ``n`` in the
    Moebius model) is the identity and is skipped.

    Vectorised over ``E``; returns shape ``(len(E), N)`` with column ``k-1``
    holding ``<A_k|A_k> = |A_{-,k}|^2 + |A_{+,k}|^2`` -- Sierra's Fig. 4 quantity,
    computed with **no** eps -> 0 / BCH approximation. ``T^{-1}`` flips the sign
    of the off-diagonal entries (``det T = 1``).
    """
    E = np.atleast_1d(np.asarray(E, dtype=np.float64))
    ell = np.asarray(ell, dtype=np.float64)
    n_max = ell.shape[0] - 1
    am = np.ones_like(E, dtype=np.complex128)
    ap = np.exp(1j * vartheta) * np.ones_like(E, dtype=np.complex128)
    norms = np.empty((E.shape[0], n_max), dtype=np.float64)
    norms[:, 0] = np.abs(am) ** 2 + np.abs(ap) ** 2
    for n in range(2, n_max + 1):
        v = varrho[n]
        if v == 0:
            norms[:, n - 1] = norms[:, n - 2]
            continue
        denom = 1.0 - abs(v) ** 2
        diag = (1.0 + abs(v) ** 2) / denom
        phase = ell[n] ** (2j * E)  # ell^{2iE}
        off01 = -2.0 * v / (denom * phase)  # -2 v ell^{-2iE} / denom
        off10 = -2.0 * np.conj(v) * phase / denom  # -2 conj(v) ell^{2iE} / denom
        am, ap = diag * am + off01 * ap, off10 * am + diag * ap
        norms[:, n - 1] = np.abs(am) ** 2 + np.abs(ap) ** 2
    return norms


def bch_amplitude_norms(
    E: np.ndarray | float,
    varrho: np.ndarray,
    ell: np.ndarray,
    *,
    vartheta: float = 0.0,
) -> np.ndarray:
    """Semiclassical (BCH) per-interval norm ``<A^T_k|A^T_k>`` (Sierra eq. 128).

    The eps -> 0 approximation to :func:`amplitude_norms` over the **same** mirror
    set. Linearising the eq.-115 matrix gives the BCH exponent
    ``W_k = sum_{n=2..k} 2 varrho_n ell_n^{-2iE}`` (the factor 2 is the eq.-115
    off-diagonal, eq. 121); writing ``R_k = |W_k|``, ``phi_k = arg(W_k)``,

        <A^T_k|A^T_k> = e^{2R_k}(1 - cos(vartheta - phi_k))
                      + e^{-2R_k}(1 + cos(vartheta - phi_k))      (eq. 128).

    This drops the O(eps^2) commutators that BCH neglects, so
    ``amplitude_norms - bch_amplitude_norms`` is the exact measure of *what finite
    eps adds*; the spike shows it vanishes as eps^2 (no new operator-intrinsic
    structure). Same mirror set / literal-eq.-115 convention as
    :func:`amplitude_norms`, so the two are directly comparable.
    """
    E = np.atleast_1d(np.asarray(E, dtype=np.float64))
    ell = np.asarray(ell, dtype=np.float64)
    n_max = ell.shape[0] - 1
    # BCH exponent W_k = sum_{n=2..k} 2 varrho_n ell_n^{-2iE}, vectorised over (E, n)
    phase = ell[None, 2:] ** (-2j * E[:, None])  # ell_n^{-2iE}, outer over (E, n)
    W = 2.0 * np.cumsum(varrho[None, 2:].astype(np.complex128) * phase, axis=1)
    R = np.abs(W)
    c = np.cos(np.angle(W) + vartheta)  # = cos(xi_k - vartheta), xi_k = -arg(W)
    body = np.exp(2.0 * R) * (1.0 - c) + np.exp(-2.0 * R) * (1.0 + c)
    out = np.empty((E.shape[0], n_max), dtype=np.float64)
    out[:, 0] = 2.0  # k = 1: |A_1|^2 = 2
    out[:, 1:] = body
    return out


def mobius_amplitude_norms(
    E: np.ndarray | float,
    n: int,
    *,
    eps: float,
    vartheta: float = 0.0,
    mu: np.ndarray | None = None,
    bch: bool = False,
) -> np.ndarray:
    """``<A_k|A_k>`` for the Moebius (Riemann) model: ``ell_n=sqrt(n)``,
    ``varrho_n = eps mu(n)/sqrt(n)`` (eq. 152).

    Convenience wrapper that builds the mirror arrays and calls the exact
    :func:`amplitude_norms` (or :func:`bch_amplitude_norms` if ``bch=True``).
    Mirrors sit only at square-free ``n`` (``mu(n) != 0``); ``n = 1`` is the
    boundary. This is the finite-eps object #44 is about -- contrast the
    eps -> 0 locator :func:`mobius_partial_sum`.
    """
    if mu is None:
        mu = mobius_sieve(n)
    k = np.arange(n + 1, dtype=np.float64)
    ell = np.sqrt(k)  # ell[1] = 1 (boundary), ell[n] = sqrt(n)
    varrho = np.zeros(n + 1, dtype=np.float64)
    varrho[2:] = eps * mu[2 : n + 1].astype(np.float64) / np.sqrt(k[2:])
    fn = bch_amplitude_norms if bch else amplitude_norms
    return fn(E, varrho, ell, vartheta=vartheta)


# --- Harmonic model (Sierra Appendix A) — the no-prime finite-eps control ------
#
# ell_n = e^{n/2}, varrho_n = eps (constant): integer proper times tau_n = 1
# (eq. 67), NOT log p. The one model in the paper with a genuine *finite-eps*
# spectrum: continuum bands separated by gaps, plus discrete levels E_n = 2 pi n.
# The iterated transfer matrix is the single E-dependent matrix S (eq. A11); the
# spectrum follows from whether S is elliptic (continuum) or hyperbolic (gap).
# Periods are integers => no primes: "finite-eps DOS exists" and "prime-built"
# are mutually exclusive here. This is the spike's positive control -- the exact
# product code reproduces these closed forms.


def _harmonic_g(eps: float) -> float:
    """Boost rapidity ``g`` of the harmonic step matrix: ``e^g=(1+eps)/(1-eps)``."""
    return 2.0 * np.arctanh(eps)


def harmonic_step_matrix(E: complex, eps: float) -> np.ndarray:
    """Iterated transfer matrix ``S = e^{-g sigma_x} e^{iE sigma_z/2}`` (eq. A11).

    ``S = [[cosh g e^{iE/2}, -sinh g e^{-iE/2}], [-sinh g e^{iE/2}, cosh g
    e^{-iE/2}]]`` with ``g = _harmonic_g(eps)``; ``det S = 1``,
    ``Tr S = 2 cosh g cos(E/2)``.
    """
    g = _harmonic_g(eps)
    cg, sg = np.cosh(g), np.sinh(g)
    e = np.exp(0.5j * E)
    return np.array([[cg * e, -sg / e], [-sg * e, cg / e]], dtype=np.complex128)


def harmonic_trace(E: np.ndarray | float, eps: float) -> np.ndarray:
    """``Tr S = 2 cosh g cos(E/2)`` (eq. A15). ``|Tr S| < 2`` <=> continuum."""
    g = _harmonic_g(eps)
    return 2.0 * np.cosh(g) * np.cos(0.5 * np.asarray(E, dtype=np.float64))


def harmonic_is_continuum(E: np.ndarray | float, eps: float) -> np.ndarray:
    """Continuum (elliptic ``S``) test ``|sin(E/2)| > tanh g`` (eq. A15)."""
    g = _harmonic_g(eps)
    return np.abs(np.sin(0.5 * np.asarray(E, dtype=np.float64))) > np.tanh(g)


def harmonic_gap_half_width(eps: float) -> float:
    """Band-edge offset ``delta`` with ``sin(pi delta) = 2 eps/(1+eps^2)`` (A16)."""
    return float(np.arcsin(2.0 * eps / (1.0 + eps**2)) / np.pi)


def harmonic_bands(eps: float, m_values: np.ndarray) -> list[tuple[float, float]]:
    """Continuum bands ``2 pi [m + delta, m + 1 - delta]`` (eq. A16)."""
    delta = harmonic_gap_half_width(eps)
    return [
        (2.0 * np.pi * (m + delta), 2.0 * np.pi * (m + 1 - delta))
        for m in np.asarray(m_values)
    ]


def harmonic_discrete_levels(m_values: np.ndarray) -> np.ndarray:
    """Discrete eigenvalues ``E_m = 2 pi m`` (eq. A13, ``vartheta = 0`` or ``pi``).

    These survive for every coupling ``g`` (every ``eps``), immersed in the
    continuum, and are exactly the integer proper-time analogue of the Moebius
    model's zeros -- but driven by the integers, not the primes.
    """
    return 2.0 * np.pi * np.asarray(m_values, dtype=np.float64)


def harmonic_amplitude_norms(
    E: np.ndarray | float, n: int, *, eps: float, vartheta: float = 0.0
) -> np.ndarray:
    """Exact ``<A_k|A_k>``, k = 1..n, by iterating the harmonic ``S`` (eq. A11).

    Same per-interval norm as :func:`amplitude_norms` but for the harmonic array;
    bounded in a band (continuum), growing in a gap, and -> 0 at ``E = 2 pi m``
    with ``vartheta = 0`` (a normalizable discrete state). Vectorised over ``E``.
    """
    E = np.atleast_1d(np.asarray(E, dtype=np.float64))
    g = _harmonic_g(eps)
    cg, sg = np.cosh(g), np.sinh(g)
    e = np.exp(0.5j * E)
    ei = 1.0 / e
    am = np.ones_like(E, dtype=np.complex128)
    ap = np.exp(1j * vartheta) * np.ones_like(E, dtype=np.complex128)
    norms = np.empty((E.shape[0], n), dtype=np.float64)
    norms[:, 0] = np.abs(am) ** 2 + np.abs(ap) ** 2
    for k in range(1, n):
        am, ap = cg * e * am - sg * ei * ap, -sg * e * am + cg * ei * ap
        norms[:, k] = np.abs(am) ** 2 + np.abs(ap) ** 2
    return norms
