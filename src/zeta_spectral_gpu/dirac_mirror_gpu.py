"""GPU path for Sierra's prime-driven Moebius-mirror locator (issue #25, Phase 2).

The GPU half of the forward locator (Piece A) in :mod:`dirac_mirror`. The decisive
object is the dense ``E``-grid scan of ``|M'_z(n)|`` (Sierra arXiv:1404.4252,
eq. 12.20): the numpy reference :func:`dirac_mirror.mobius_partial_sum`
materializes the full ``(len(E) x n)`` complex phase matrix, which is the memory
wall as the grid is refined and ``n`` grows. The hand-written kernel
``kernels/dirac_mirror.cu`` computes the same sum with one thread per ``E`` point
and an inner loop over ``k`` -- ``O(len(E) * n)`` work in ``O(len(E) + n)`` memory.

fp64 throughout. The locator lives in an ``O(1)``, cancellation-free regime, so
double precision suffices and reproduces the CPU reference to floating-point
tolerance (the house GPU-vs-CPU rule) -- this is the clean fp64 GPU win the #25
forward ruling anticipated, the opposite regime from the flagship's mpmath-bound
eigensolve. CuPy is imported lazily so the package imports on a CPU-only box.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import dirac_mirror

_KERNEL_SRC = Path(__file__).with_name("kernels") / "dirac_mirror.cu"


def _cupy():
    """Import cupy on demand, with a clear error if the GPU extra is missing."""
    from ._cuda_dll import add_cuda_dll_directories

    add_cuda_dll_directories()  # no-op for NVRTC; kept for parity with the GPU shims
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
    """Compile (once) and return the RawModule for dirac_mirror.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


def _amp_logk(
    n: int,
    sigma: float,
    weights: np.ndarray | None,
    mu: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side fp64 per-term arrays ``amp[k] = c(k) k^{-sigma}`` and ``log k``.

    Mirrors the precompute in :func:`dirac_mirror.mobius_partial_sum`: ``c(k)``
    defaults to ``mu(k)`` (zeta); pass ``weights = chi(k) mu(k)`` for a Dirichlet
    ``L``-function. ``amp`` keeps the weights' dtype -- real for zeta and
    real characters (the fast ``mobius_locator`` path), ``complex128`` for a
    genuinely complex character (the ``weighted_locator`` path). This is the only
    number-theoretic input the kernel consumes (forward, not inverse), and it is
    ``O(n)`` -- cheap next to the ``O(len(E)*n)`` kernel, so it stays on the host.
    """
    k = np.arange(1, n + 1, dtype=np.float64)
    if weights is None:
        if mu is None:
            mu = dirac_mirror.mobius_sieve(n)
        weights = mu[1 : n + 1].astype(np.float64)
    else:
        weights = np.asarray(weights[:n])  # preserve real/complex dtype (L-functions)
    return weights * k**-sigma, np.log(k)


def mobius_partial_sum_gpu(
    E: np.ndarray | float,
    n: int,
    *,
    sigma: float = 0.5,
    weights: np.ndarray | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray | complex:
    """``M'_z(n) = sum_{k<=n} c(k) k^{-(sigma+iE)}`` on the GPU -- the locator scan.

    Kernel-backed mirror of :func:`dirac_mirror.mobius_partial_sum` (Piece A): one
    thread per ``E``-grid point sums over all ``k``, so the ``(len(E) x n)`` phase
    matrix the numpy reference forms is never materialized. fp64 ``complex128``;
    returns an array shaped like ``E`` (a Python ``complex`` for scalar ``E``), and
    agrees with the CPU reference to floating-point tolerance (the house rule).

    No boundary phase ``vartheta`` and no zeros enter -- the zeros are read *off*
    of ``|M'_z|`` downstream, never fed in.
    """
    cp = _cupy()
    E_arr = np.asarray(E, dtype=np.float64)
    E_flat = np.ascontiguousarray(np.atleast_1d(E_arr).ravel())
    amp, logk = _amp_logk(n, sigma, weights, mu)

    logk_d = cp.asarray(logk)
    E_d = cp.asarray(E_flat)
    n_e = int(E_flat.size)
    out_re = cp.empty(n_e, dtype=cp.float64)
    out_im = cp.empty(n_e, dtype=cp.float64)
    threads = 256
    blocks = (n_e + threads - 1) // threads

    if np.iscomplexobj(amp):
        # Complex character (Dirichlet L): one extra real array for Im(amp).
        amp_re = cp.asarray(np.ascontiguousarray(amp.real))
        amp_im = cp.asarray(np.ascontiguousarray(amp.imag))
        kernel = _module().get_function("weighted_locator")
        args = (amp_re, amp_im, logk_d, np.int64(n), E_d, np.int64(n_e), out_re, out_im)
    else:
        # Real coefficients (zeta, principal/quadratic characters): fast path.
        amp_d = cp.asarray(np.ascontiguousarray(amp))
        kernel = _module().get_function("mobius_locator")
        args = (amp_d, logk_d, np.int64(n), E_d, np.int64(n_e), out_re, out_im)
    kernel((blocks,), (threads,), args)

    out = cp.asnumpy(out_re) + 1j * cp.asnumpy(out_im)
    if E_arr.ndim == 0:
        return complex(out[0])
    return out.reshape(E_arr.shape)
