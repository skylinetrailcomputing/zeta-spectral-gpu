"""GPU dense eigensolve for Sierra's deformed-``xp`` operator (issue #31).

The GPU half of #23. The CPU secular reference :func:`deformed_xp.secular_spectrum`
(real roots of the Bessel-K equation, eq. 14) is the truth spectrum; this module
assembles the operator as a dense Hermitian matrix and diagonalises it with
``cupy.linalg.eigh`` (cuSOLVER), reproducing those roots to floating-point
precision for the resolved low modes — the house GPU-vs-CPU agreement rule.

Why a Galerkin assembly (spike findings, PR #30): the operator (Sierra eq. 10,
``hbar = 1``)

    H psi(x) = -i (x psi'(x) + 1/2 psi(x)) - i l_p^2 sqrt(x) int_x^inf sqrt(y) psi(y) dy

is first-order *and* nonlocal on ``x >= l_x``, self-adjoint only via the nonlocal
boundary condition (eq. 11) with extension angle ``theta``,

    e^{i theta} sqrt(l_x) psi(l_x) + l_p int_{l_x}^inf sqrt(x) psi(x) dx = 0.

A naive uniform grid yields a spurious particle-in-a-box spectrum (cutoff
dependent, below the classical floor ``|E| >= 2h = 4 pi``). Instead we use a
**decaying Galerkin basis** of scaled Laguerre functions on ``[l_x, inf)``,

    phi_k(x) = sqrt(c) L_k(c t) e^{-c t / 2},   t = x - l_x,

matching the eigenfunction decay ``psi_E ~ e^{-l_p x}`` (eq. 13) so there are no
box modes to begin with, and they are orthonormal (``int phi_m phi_n dx = delta``).

**Assembly.** The sesquilinear form ``<phi_m, H phi_n> = i M_mn`` with the real

    M_mn = l_x phi_m(l_x) phi_n(l_x) + Q_mn + 1/2 delta_mn - l_p^2 K_mn,

where ``Q_mn = int x phi_m' phi_n dx`` (local) and ``K_mn = int sqrt(x) phi_m I_n dx``
with ``I_n(x) = int_x^inf sqrt(y) phi_n dy`` (nonlocal). All elements are computed
with Gauss-Laguerre quadrature matched to the basis decay; the nonlocal tail
``I_n(x)`` uses an inner Gauss-Laguerre rule (``y = x + 2 tau / c``) rather than a
masked node sum, which is what makes the convergence spectral.

**The self-adjoint extension.** The symmetry defect of ``H`` is exactly the
boundary form ``B(phi, psi) = i[l_x phi(l_x)psi(l_x) - l_p^2 I_phi(l_x) I_psi(l_x)]``
(rank two), which vanishes identically on the null space of the eq. 11 functional
``g_n = e^{i theta} sqrt(l_x) phi_n(l_x) + l_p I_n(l_x)``. Projecting ``i M`` onto
``null(g)`` therefore gives an *exactly Hermitian* reduced matrix whose spectrum is
the ``+-E`` time-conjugate pairs (eq. 14, ``theta != pi``); the positive
eigenvalues are the average-zero analogues that match the secular roots.

Forward, not inverse: a geometric deformation of ``xp`` — no primes, no zeros
consumed. See :mod:`zeta_spectral_gpu.deformed_xp` (#23) and ``knowledge/``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_genlaguerre, eval_laguerre, roots_laguerre

from .deformed_xp import BESSEL_ARG, CLASSICAL_BOUND, THETA_RIEMANN

# Scaling symmetry: only the product h = l_x l_p = BESSEL_ARG (= 2 pi) enters the
# spectrum, so we fix l_x = 1 and l_p = h and assemble on [1, inf).
L_X = 1.0
L_P = BESSEL_ARG


def _cupy():
    """Import cupy on demand (cuSOLVER DLLs wired up first on Windows)."""
    from ._cuda_dll import add_cuda_dll_directories

    add_cuda_dll_directories()
    try:
        import cupy as cp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cupy is required for the GPU eigensolve. Install the wheel matching "
            "your CUDA runtime, e.g. `uv sync --extra gpu`."
        ) from exc
    return cp


def _tail_stripped(s_at: np.ndarray, c: float, n: int, n_inner: int) -> np.ndarray:
    """Weight-stripped nonlocal tail ``I~_n(x) = e^{c(x-l_x)/2} int_x^inf sqrt(y) phi_n``.

    Substituting ``y = x + 2 tau / c`` turns the ``e^{-c(y-l_x)/2}`` decay into the
    Gauss-Laguerre weight ``e^{-tau}``; the leading ``e^{-c(x-l_x)/2}`` is stripped
    so no growing exponential is ever formed (overflow-safe at high order). Returns
    an ``(n, len(s_at))`` array indexed by basis order then evaluation point, where
    ``s_at = c (x - l_x)``.
    """
    tau, om = roots_laguerre(n_inner)
    s_at = np.atleast_1d(np.asarray(s_at, dtype=np.float64))
    arg = s_at[:, None] + 2.0 * tau[None, :]  # c (y - l_x) at the inner nodes
    base = om[None, :] * np.sqrt(L_X + arg / c)  # (P, n_inner)
    scale = np.sqrt(c) * (2.0 / c)
    out = np.empty((n, s_at.size), dtype=np.float64)
    for k in range(n):
        out[k] = scale * np.einsum("pl,pl->p", base, eval_laguerre(k, arg))
    return out


def galerkin_matrix(
    n: int,
    *,
    theta: float = THETA_RIEMANN,
    basis_scale: float | None = None,
    n_quad: int | None = None,
    n_inner: int = 80,
) -> np.ndarray:
    """Assemble the reduced Hermitian matrix of the deformed-``xp`` operator.

    Builds the ``n``-function scaled-Laguerre Galerkin matrix ``i M`` and projects
    it onto the null space of the eq. 11 boundary functional for extension angle
    ``theta``, returning the ``(n-1, n-1)`` complex Hermitian matrix whose
    eigenvalues are the ``+-E`` pairs. ``basis_scale`` is the Laguerre rate ``c``
    (default ``4 l_p``, a good resolution/conditioning tradeoff); ``n_quad`` is the
    outer Gauss-Laguerre order (default ``3 n``); ``n_inner`` the inner order for
    the nonlocal tail.
    """
    c = 4.0 * L_P if basis_scale is None else float(basis_scale)
    m = max(60, 3 * n) if n_quad is None else int(n_quad)

    s, w = roots_laguerre(m)  # bare weights: integrands are (smooth) * e^{-s}
    x = L_X + s / c
    sc = np.sqrt(c)
    ln = np.array([eval_laguerre(k, s) for k in range(n)])
    ln1 = np.array(
        [eval_genlaguerre(k - 1, 1, s) if k else np.zeros_like(s) for k in range(n)]
    )

    overlap = np.einsum("i,mi,ni->mn", w, ln, ln)  # int phi_m phi_n dx (= I)
    dpoly = -ln1 - 0.5 * ln  # phi_m'(x) = sc c dpoly e^{-s/2}
    local = c * np.einsum("i,i,mi,ni->mn", w, x, dpoly, ln)  # int x phi_m' phi_n

    tail = _tail_stripped(s, c, n, n_inner)  # I~_n(x_i)  (n, m)
    nonlocal_ = (sc / c) * np.einsum("i,i,mi,ni->mn", w, np.sqrt(x), ln, tail)
    tail_at_lx = _tail_stripped(0.0, c, n, n_inner)[:, 0]  # I_n(l_x)

    phi_lx = sc  # phi_k(l_x) = sqrt(c) for all k
    boundary = L_X * phi_lx * phi_lx
    m_real = boundary + local + 0.5 * overlap - L_P**2 * nonlocal_
    g = np.exp(1j * theta) * np.sqrt(L_X) * phi_lx + L_P * tail_at_lx

    # Project i M onto null(g): the boundary defect vanishes there, so the reduced
    # matrix is exactly Hermitian (symmetrise only the round-off residual).
    null = np.linalg.svd(g.reshape(1, n))[2].conj().T[:, 1:]  # (n, n-1)
    h_red = null.conj().T @ (1j * m_real) @ null
    return 0.5 * (h_red + h_red.conj().T)


def eigenvalues(matrix: np.ndarray, *, use_gpu: bool = True) -> np.ndarray:
    """Eigenvalues of a Hermitian matrix via ``cupy.linalg.eigh`` (cuSOLVER).

    Falls back to ``numpy.linalg.eigvalsh`` (the small-N CPU reference) when the
    GPU is unavailable or ``use_gpu=False``. Returned ascending.
    """
    if use_gpu:
        try:
            cp = _cupy()
            return cp.asnumpy(cp.linalg.eigvalsh(cp.asarray(matrix)))
        except ImportError:
            pass
    return np.linalg.eigvalsh(matrix)


def spectrum(
    n: int,
    *,
    theta: float = THETA_RIEMANN,
    basis_scale: float | None = None,
    n_quad: int | None = None,
    n_inner: int = 80,
    use_gpu: bool = True,
    tol: float = 1e-6,
) -> np.ndarray:
    """Positive eigenvalues of the deformed-``xp`` operator (the average-zero analogues).

    Assembles the Galerkin matrix (:func:`galerkin_matrix`) and diagonalises it
    (:func:`eigenvalues`), returning the positive half of the ``+-E`` spectrum,
    ascending — the GPU counterpart of :func:`deformed_xp.secular_spectrum`. Only
    the lower portion is resolved to floating-point precision (spectral
    convergence); request ``n`` comfortably larger than the number of levels
    needed.
    """
    h = galerkin_matrix(
        n, theta=theta, basis_scale=basis_scale, n_quad=n_quad, n_inner=n_inner
    )
    eig = eigenvalues(h, use_gpu=use_gpu)
    return np.sort(eig[eig > tol])


__all__ = [
    "CLASSICAL_BOUND",
    "L_P",
    "L_X",
    "eigenvalues",
    "galerkin_matrix",
    "spectrum",
]
