"""GPU / fp64 paths for the Connes–Consani–Moscovici operator (issue #9).

This is the *throughput* companion to the multiprecision reference in
:mod:`ccm` (issue #8). Two things move here:

1. **The special-function matrix fill**, in fp64. The per-mode archimedean and
   prime coefficients are computed in double precision on the host
   (:func:`_per_mode_arrays_fp64`); the ``O(N^2)`` divided-difference assembly —
   the hot path — runs in a hand-written CUDA kernel (``kernels/ccm_assembly.cu``)
   loaded via CuPy ``RawModule``. :func:`assemble_weil_matrix_fp64` is the pure-numpy
   CPU reference the kernel is tested against, exactly mirroring the
   ``spacing.py`` / ``spacing_gpu.py`` / ``kernels/spacing.cu`` triple.

2. **An fp64 conditioning probe** (:func:`conditioning_fp64`) via ``cupy.linalg.eigh``
   (cuSOLVER), used to make the precision wall *empirical*.

**The precision wall (why the eigensolve stays on CPU).** The minimal eigenvalue
``epsilon_N`` of ``QW_lambda^N`` drops below fp64 epsilon almost immediately — at
``N = 50`` it is ``~1e-17`` already at the cutoff ``x = 5`` and reaches ``~1e-61``
by ``x = 14``. So an fp64 eigensolve recovers the near-null eigenvector (hence the
operator spectrum and the convergence/conditioning curves) *only* for the
smallest cutoffs; beyond that it returns roundoff. That is the whole reason the
extended-precision eigensolve lives in :mod:`ccm` (mpmath), and the GPU's role
here is the fill, the fp64-regime statistics, and demonstrating exactly where
fp64 falls off. See ``CLAUDE.md`` (precision reality) and issue #9.

Everything below mirrors :func:`ccm.assemble_weil_matrix` in fp64; the equation
numbers refer to ``knowledge/ccm-operator.md`` as in :mod:`ccm`.
"""

from __future__ import annotations

import functools
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ccm

PI = np.pi
_KERNEL_SRC = Path(__file__).with_name("kernels") / "ccm_assembly.cu"


# ----------------------------------------------------------------------------
# CuPy / CUDA-library bootstrap
# ----------------------------------------------------------------------------
#
# cuSOLVER (used by cupy.linalg.eigh) loads its DLLs lazily on first use. With
# the pip-wheel CUDA libraries (cupy-cuda12x[ctk]) those DLLs sit under
# site-packages/nvidia/*/bin, which is NOT on the Windows DLL search path, so the
# first eigh raises "DLL load failed importing cusolver". Adding the wheel bin
# dirs via os.add_dll_directory fixes it. NVRTC (the RawModule path) already
# resolves via cuda-pathfinder, so spacing_gpu never needed this; the eigensolve
# does.

_CUDA_LIBS_READY = False


def _ensure_cuda_libs() -> None:
    """Put the pip-wheel CUDA library dirs on the DLL search path (Windows)."""
    global _CUDA_LIBS_READY
    if _CUDA_LIBS_READY:
        return
    if os.name == "nt":
        for p in sys.path:
            base = os.path.join(p, "nvidia")
            if os.path.isdir(base):
                for d in glob.glob(os.path.join(base, "*", "bin")):
                    try:
                        os.add_dll_directory(d)
                    except OSError:  # pragma: no cover - defensive
                        pass
    _CUDA_LIBS_READY = True


def _cupy():
    """Import cupy on demand (with the CUDA-lib bootstrap), clear error if absent."""
    _ensure_cuda_libs()
    try:
        import cupy as cp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cupy is required for GPU paths. Install the wheel matching your "
            "CUDA runtime, e.g. `uv sync --extra gpu`."
        ) from exc
    return cp


@functools.lru_cache(maxsize=1)
def _module():
    """Compile (once) and return the RawModule for ccm_assembly.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


# ----------------------------------------------------------------------------
# fp64 special functions
# ----------------------------------------------------------------------------
#
# The archimedean entries (Prop. 4.2) need a Lerch transcendent and the
# digamma / trigamma of a complex argument. mpmath supplies these at any
# precision but is slow and arbitrary-precision; here we want fp64, vectorized
# over the 2N+1 mode indices. Each is validated against mpmath in the tests.


def _lerch_fp64(q: float, s: int, a, *, max_terms: int = 400):
    """Lerch transcendent ``Phi(q, s, a) = sum_{j>=0} q^j / (a + j)^s`` in fp64.

    Vectorized over ``a`` (real or complex array). The only argument we ever pass
    is ``q = e^{-2L} < 1`` (the source's geometric series parameter), so the sum
    converges in a handful of terms and a direct accumulation is both fast and
    accurate — the same reduction :func:`ccm._lerch` uses in mpmath.
    """
    a = np.asarray(a)
    out = np.zeros(
        a.shape,
        dtype=np.result_type(a.dtype, np.float64, complex)
        if np.iscomplexobj(a)
        else np.result_type(a.dtype, np.float64),
    )
    qj = 1.0
    for j in range(max_terms):
        out = out + qj / (a + j) ** s
        qj *= q
        if qj < 1e-19:
            break
    return out


# Asymptotic (Stirling) coefficients for digamma/trigamma: enough terms that the
# series is at the fp64 floor once the recurrence has pushed Re(z) past ~14.
_DIGAMMA_BERN = (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760)
_TRIGAMMA_BERN = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730)


def _digamma_c(z, *, n_rec: int = 14):
    """Digamma ``psi(z)`` for complex ``z`` (fp64), via recurrence + asymptotics.

    ``psi(z) = psi(z + n_rec) - sum_{k<n_rec} 1/(z+k)``; the shifted argument has
    ``Re >= 14``, where the Stirling tail ``psi(w) ~ ln w - 1/(2w) - sum B_2k/(2k)
    w^{-2k}`` is at the fp64 floor. Vectorized over ``z``.
    """
    z = np.asarray(z, dtype=complex)
    s = np.zeros_like(z)
    w = z.copy()
    for _ in range(n_rec):
        s -= 1.0 / w
        w = w + 1.0
    inv = 1.0 / w
    inv2 = inv * inv
    res = np.log(w) - 0.5 * inv
    term = inv2  # w^{-2}; non-in-place *= below so inv2 is never mutated
    for c in _DIGAMMA_BERN:
        res = res - c * term
        term = term * inv2
    return res + s


def _trigamma_c(z, *, n_rec: int = 14):
    """Trigamma ``psi'(z)`` for complex ``z`` (fp64), via recurrence + asymptotics.

    ``psi'(z) = sum_{k<n_rec} 1/(z+k)^2 + psi'(z + n_rec)`` with the Stirling tail
    ``psi'(w) ~ 1/w + 1/(2w^2) + sum B_2k w^{-(2k+1)}``. Vectorized over ``z``.
    """
    z = np.asarray(z, dtype=complex)
    s = np.zeros_like(z)
    w = z.copy()
    for _ in range(n_rec):
        s += 1.0 / (w * w)
        w = w + 1.0
    inv = 1.0 / w
    inv2 = inv * inv
    res = inv + 0.5 * inv2
    term = inv * inv2  # w^{-3}
    for c in _TRIGAMMA_BERN:
        res = res + c * term
        term = term * inv2
    return res + s


_DIGAMMA_QUARTER = float(_digamma_c(np.array([0.25 + 0j]))[0].real)


# ----------------------------------------------------------------------------
# fp64 per-mode coefficients (the O(N) special-function part)
# ----------------------------------------------------------------------------


@dataclass
class _PerMode:
    """The fp64 arrays the assembly consumes (see :func:`assemble_weil_matrix_fp64`)."""

    B: np.ndarray  # length 2N+1, odd: J(n) + prime sin-sum
    diag_full: np.ndarray  # length 2N+1, even: arch diagonal + prime cos-sum
    a: np.ndarray  # length 2N+1, point-mass coefficient 1/(L^2 + 16 pi^2 n^2)
    L: float
    pref: float  # point-mass prefactor 32 L sinh(L/4)^2


def _per_mode_arrays_fp64(N: int, lam: float) -> _PerMode:
    """Compute the fp64 per-mode coefficient arrays for ``QW_lambda^N``.

    Mirrors the precompute block of :func:`ccm.assemble_weil_matrix`: the odd
    ``B[n] = J(n) + sum_k coef_k sin(2 pi n log k / L)`` and the even diagonal
    ``diag[n] = (arch diagonal) + sum_k coef_k * 2(1 - log k / L) cos(2 pi n log k / L)``,
    plus the point-mass coefficients ``a[n]``. The primes are the same von
    Mangoldt input as the reference (``ccm.prime_powers``) — forward, not inverse.
    """
    lam = float(lam)
    L = 2.0 * np.log(lam)
    q = np.exp(-2.0 * L)
    dim = 2 * N + 1

    # Prime input: same enumeration as the mpmath reference, cast to fp64.
    pps = ccm.prime_powers(lam**2)
    ks = np.array([k for k, _ in pps], dtype=np.float64)
    weights = np.array([float(w) for _, w in pps])  # Lambda(k) = log p
    if ks.size:
        logs_k = np.log(ks)
        coefs = weights / np.sqrt(ks)
    else:  # no primes <= x (only when lam^2 < 2)
        logs_k = np.empty(0)
        coefs = np.empty(0)

    n = np.arange(0, N + 1, dtype=np.float64)
    zk = 0.25 + 1j * (PI * n / L)  # z_k = i pi k / L + 1/4, k = 0..N

    # archimedean J, K, M via the closed forms (eqs. 4.5-4.7), fp64.
    em = np.exp(-L / 2.0)
    hyp = zk * _lerch_fp64(q, 1, zk)  # hyp2f1(1, z, z+1, q)

    coef_j = 2.0 * L / (L + 4.0j * PI * n)
    J = em * np.imag(coef_j * hyp) + np.imag(_digamma_c(zk)) / 2.0

    k1 = -L * em * np.imag((2.0 * L / (4.0 * PI * n - 1j * L)) * hyp)
    k2 = -(em / 4.0) * np.real(_lerch_fp64(q, 2, zk))
    k3 = np.real(_trigamma_c(zk)) / 4.0
    Karr = k1 + k2 + k3

    m1 = -em * np.real((2.0 * L / (L + 4.0j * PI * n)) * hyp)
    m2 = 2.0 * em * 0.25 * _lerch_fp64(q, 1, np.array([0.25 + 0j]))[0].real
    m3 = -np.real(_digamma_c(zk) - _DIGAMMA_QUARTER) / 2.0
    Marr = m1 + m2 + m3

    diag_const = _archimedean_diag_const_fp64(L, q)

    # Prime sin/cos sums per mode (vectorized over k).
    if coefs.size:
        wk = 2.0 * PI * np.outer(n, logs_k) / L  # (N+1, #primes)
        sin_sum = (np.sin(wk) * coefs).sum(axis=1)
        cos_sum = (np.cos(wk) * (coefs * 2.0 * (1.0 - logs_k / L))).sum(axis=1)
    else:
        sin_sum = np.zeros(N + 1)
        cos_sum = np.zeros(N + 1)

    b_pos = J + sin_sum
    diag_pos = diag_const + 2.0 * Marr - (2.0 / L) * Karr + cos_sum

    # Mirror to all modes i = 0..2N (B odd, diag even).
    idx = np.arange(dim) - N
    B = np.where(idx < 0, -b_pos[np.abs(idx)], b_pos[np.abs(idx)])
    diag_full = diag_pos[np.abs(idx)]

    a = 1.0 / (L**2 + 16.0 * PI**2 * idx.astype(np.float64) ** 2)
    pref = 32.0 * L * np.sinh(L / 4.0) ** 2
    return _PerMode(B=B, diag_full=diag_full, a=a, L=L, pref=float(pref))


def _archimedean_diag_const_fp64(L: float, q: float) -> float:
    """The mode-independent diagonal constant ``G + 2 C0`` (fp64; cf. :func:`ccm`)."""
    g = np.euler_gamma + np.log(4.0 * PI * (np.exp(L) - 1.0) / (np.exp(L) + 1.0))
    quarter = np.array([0.25 + 0j])
    half = np.array([0.5 + 0j])
    c0 = (
        np.log(2.0) / 2.0
        + PI / 4.0
        - np.exp(-L / 2.0) / 2.0 * _lerch_fp64(q, 1, quarter)[0].real
        + np.exp(-L) / 2.0 * _lerch_fp64(q, 1, half)[0].real
    )
    return float(g + 2.0 * c0)


# ----------------------------------------------------------------------------
# Assembly: numpy reference + CUDA kernel
# ----------------------------------------------------------------------------


def assemble_weil_matrix_fp64(N: int, lam: float) -> np.ndarray:
    """Assemble ``QW_lambda^N`` in fp64 with numpy — the CPU reference for the GPU.

    Same sign pattern as :func:`ccm.assemble_weil_matrix`
    (``QW = W_{0,2} - W_R - sum_p W_p``), built by broadcasting from the per-mode
    arrays: the off-diagonal is the Lemma 5.1 divided difference
    ``(B[j] - B[i]) / (pi (n - m))``, the diagonal carries ``diag_full``, and the
    point mass adds ``pref (L^2 - 16 pi^2 n m) a_i a_j``.
    """
    pm = _per_mode_arrays_fp64(N, lam)
    dim = 2 * N + 1
    n = np.arange(-N, N + 1, dtype=np.float64)

    point_mass = (
        pm.pref * (pm.L**2 - 16.0 * PI**2 * np.outer(n, n)) * np.outer(pm.a, pm.a)
    )

    diff = pm.B[None, :] - pm.B[:, None]  # B[j] - B[i]
    denom = PI * (n[:, None] - n[None, :])  # pi (n - m) = pi (i - j)
    off = np.zeros((dim, dim))
    np.divide(diff, denom, out=off, where=denom != 0.0)

    A = point_mass - off
    A[np.diag_indices(dim)] = np.diag(point_mass) - pm.diag_full
    return A


def assemble_weil_matrix_gpu(N: int, lam: float):
    """Assemble ``QW_lambda^N`` in fp64 on the GPU via ``kernels/ccm_assembly.cu``.

    Returns a CuPy ``(2N+1, 2N+1)`` float64 array (left on the device so the
    conditioning probe can run ``eigh`` without a round-trip). The per-mode
    coefficients are computed on the host (:func:`_per_mode_arrays_fp64`); the
    kernel does the ``O(N^2)`` divided-difference fill, one thread per entry —
    the deliberate hand-written-CUDA surface (cf. ``kernels/spacing.cu``).
    """
    cp = _cupy()
    pm = _per_mode_arrays_fp64(N, lam)
    dim = 2 * N + 1
    B = cp.asarray(pm.B)
    a = cp.asarray(pm.a)
    diag = cp.asarray(pm.diag_full)
    A = cp.empty(dim * dim, dtype=cp.float64)

    kernel = _module().get_function("ccm_assemble")
    threads = 256
    blocks = (dim * dim + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            B,
            a,
            diag,
            np.int64(N),
            np.float64(pm.pref),
            np.float64(pm.L**2),
            np.float64(16.0 * PI**2),
            np.float64(PI),
            A,
        ),
    )
    return A.reshape(dim, dim)


# ----------------------------------------------------------------------------
# fp64 conditioning probe (the precision-wall demonstration)
# ----------------------------------------------------------------------------


@dataclass
class Conditioning:
    """fp64 conditioning diagnostics of ``QW_lambda^N`` (via cuSOLVER ``eigh``).

    ``eps_n`` is the smallest ``|eigenvalue|`` — the finite-cutoff near-null mode.
    It is faithful only while it stays above the fp64 floor (~1e-13 for these
    matrices); below that ``eps_n`` and ``cond`` saturate at roundoff, which is
    precisely the precision wall #9 documents.
    """

    eps_n: float
    cond: float
    eig_max: float


def conditioning_fp64(A) -> Conditioning:
    """Smallest ``|eigenvalue|`` and condition number of ``A`` via fp64 ``eigh``.

    Accepts a numpy or CuPy matrix; uses ``cupy.linalg.eigh`` (cuSOLVER). This is
    the *conditioning probe*, not the spectrum-producing eigensolve — that one is
    precision-bound and lives in :mod:`ccm` (mpmath).
    """
    cp = _cupy()
    Ag = cp.asarray(A)
    w = cp.abs(cp.linalg.eigh(Ag)[0])
    eps_n = float(cp.min(w))
    eig_max = float(cp.max(w))
    return Conditioning(eps_n=eps_n, cond=eig_max / eps_n, eig_max=eig_max)
