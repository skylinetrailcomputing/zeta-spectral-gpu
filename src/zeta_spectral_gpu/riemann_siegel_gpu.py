"""GPU path for the Riemann-Siegel Hardy-Z / zeta evaluator (issue #55, GPU half).

The decisive cost in :mod:`riemann_siegel` is the main sum -- ``O(sqrt(t))`` cosine
terms per height. Evaluating ``Z`` (or ``zeta``) on a dense grid of heights, or at
one very large height, is therefore embarrassingly parallel: the hand-written
kernel ``kernels/riemann_siegel.cu`` gives one thread per height, looping over its
own ``N = floor(sqrt(t/2pi))`` terms, so the ``(n_e x N)`` work never materializes
as a matrix (the numpy reference :func:`riemann_siegel.main_sum` loops over ``n``
with a per-height mask instead -- same arithmetic, host-side).

The two ``O(n_e)`` pieces -- the Riemann-Siegel ``theta`` phase and the asymptotic
remainder ``C_0 + C_1 + C_2`` -- stay on the host in exact fp64 (no ``mpmath``;
see :func:`riemann_siegel.rs_remainder`), so the GPU carries the ``O(n_e * N)``
arithmetic and the result still reproduces the CPU reference to floating-point
tolerance (the house GPU-vs-CPU rule). fp64 throughout: the main sum is a
cancellation-free ``O(1)`` object at the heights of interest, the clean fp64 GPU
win (contrast the flagship's mpmath-bound eigensolve). CuPy is imported lazily so
the package imports on a CPU-only box.

Forward, not inverse: a tool that evaluates ``zeta`` from its Riemann-Siegel
expansion -- no zeros consumed (see :mod:`riemann_siegel`).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import riemann_siegel

_KERNEL_SRC = Path(__file__).with_name("kernels") / "riemann_siegel.cu"


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
    """Compile (once) and return the RawModule for riemann_siegel.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


def main_sum_gpu(t: np.ndarray | float) -> np.ndarray | float:
    """Riemann-Siegel main sum on the GPU -- kernel-backed mirror of
    :func:`riemann_siegel.main_sum`.

    One thread per height sums its own ``N = floor(sqrt(t/2pi))`` terms; the
    ``theta`` phase is precomputed on the host. Returns an array shaped like ``t``
    (a Python ``float`` for scalar ``t``) and agrees with the CPU reference to
    floating-point tolerance.
    """
    cp = _cupy()
    t_arr = np.asarray(t, dtype=np.float64)
    t_flat = np.ascontiguousarray(np.atleast_1d(t_arr).ravel())
    th = np.ascontiguousarray(np.atleast_1d(riemann_siegel.theta(t_flat)))
    big_n = np.floor(np.sqrt(t_flat / riemann_siegel.TWO_PI)).astype(np.int64)
    n_max = int(big_n.max()) if big_n.size else 0

    k = np.arange(1, n_max + 1, dtype=np.float64)
    amp = 1.0 / np.sqrt(k)
    logn = np.log(k)

    amp_d = cp.asarray(amp)
    logn_d = cp.asarray(logn)
    t_d = cp.asarray(t_flat)
    th_d = cp.asarray(th)
    n_d = cp.asarray(big_n)
    n_e = int(t_flat.size)
    out = cp.empty(n_e, dtype=cp.float64)

    threads = 256
    blocks = (n_e + threads - 1) // threads
    kernel = _module().get_function("rs_main_sum")
    kernel((blocks,), (threads,), (amp_d, logn_d, t_d, th_d, n_d, np.int64(n_e), out))

    res = cp.asnumpy(out)
    if t_arr.ndim == 0:
        return float(res[0])
    return res.reshape(t_arr.shape)


def hardy_z_gpu(
    t: np.ndarray | float, *, correction_terms: int = 2
) -> np.ndarray | float:
    """Hardy ``Z(t)`` on the GPU: kernel main sum + host fp64 remainder.

    Kernel-backed mirror of :func:`riemann_siegel.hardy_z`. The GPU computes the
    ``O(n_e * N)`` main sum; the host adds the ``O(n_e)`` asymptotic remainder
    (:func:`riemann_siegel.rs_remainder`, pure fp64), so the result is bit-for-bit
    the same formula as the CPU path and matches it to floating-point tolerance.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    ms = np.atleast_1d(main_sum_gpu(t_arr))
    rem = np.atleast_1d(
        riemann_siegel.rs_remainder(t_arr, correction_terms=correction_terms)
    )
    z = ms + rem
    if t_arr.ndim == 0:
        return float(z[0])
    return z.reshape(t_arr.shape)


def zeta_critical_gpu(
    t: np.ndarray | float, *, correction_terms: int = 2
) -> np.ndarray | complex:
    """``zeta(1/2 + i t) = Z(t) e^{-i theta(t)}`` on the GPU (fp64 complex).

    Kernel-backed mirror of :func:`riemann_siegel.zeta_critical`.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    z = np.atleast_1d(hardy_z_gpu(t_arr, correction_terms=correction_terms))
    out = z * np.exp(-1j * np.atleast_1d(riemann_siegel.theta(t_arr)))
    if t_arr.ndim == 0:
        return complex(out[0])
    return out.reshape(t_arr.shape)
