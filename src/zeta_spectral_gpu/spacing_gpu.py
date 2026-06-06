"""GPU statistics via CuPy + the hand-written kernels in ``kernels/spacing.cu``.

These mirror ``spacing.py`` exactly; they exist to run the same reductions at
large N. CuPy is imported lazily so the package imports on machines without a
GPU (and without the optional ``gpu`` extra installed).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import spacing

_KERNEL_SRC = Path(__file__).with_name("kernels") / "spacing.cu"


def _cupy():
    """Import cupy on demand, with a clear error if the GPU extra is missing."""
    from ._cuda_dll import add_cuda_dll_directories

    add_cuda_dll_directories()  # Windows: nvidia/*/bin on the DLL path (cuSOLVER)
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
    """Compile (once) and return the RawModule for spacing.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


def pair_correlation_histogram_gpu(
    unfolded: np.ndarray, bin_width: float, max_sep: float
) -> np.ndarray:
    """GPU pair-correlation histogram. Must match the numpy reference in spacing.py."""
    cp = _cupy()
    x = cp.sort(cp.asarray(unfolded, dtype=cp.float64))
    n = int(x.size)
    n_bins = int(np.ceil(max_sep / bin_width))
    hist = cp.zeros(n_bins, dtype=cp.uint64)

    kernel = _module().get_function("pair_correlation_hist")
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            x,
            np.int64(n),
            np.float64(bin_width),
            np.int64(n_bins),
            np.float64(max_sep),
            hist,
        ),
    )
    return cp.asnumpy(hist).astype(np.int64)


def nearest_neighbour_spacings_gpu(unfolded: np.ndarray) -> np.ndarray:
    """GPU consecutive spacings via the consecutive_diff kernel."""
    cp = _cupy()
    x = cp.asarray(unfolded, dtype=cp.float64)
    n = int(x.size)
    out = cp.empty(n - 1, dtype=cp.float64)

    kernel = _module().get_function("consecutive_diff")
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), (x, np.int64(n), out))
    return cp.asnumpy(out)


# --- Long-range rigidity (Sigma^2, Delta_3) ---------------------------------
#
# No hand-written kernel here: the workload is a vectorized binary search
# (``searchsorted``), which CuPy provides as a fast GPU primitive — the same
# reason eigensolves stay in cuSOLVER. These mirror ``spacing.number_variance`` /
# ``spacing.dyson_mehta_delta3`` and use identical window starts so the GPU-vs-CPU
# test agrees to floating-point tolerance. Delta_3 reuses the cheap host-side
# Mehta transform (``spacing.delta3_from_sigma2``) on a GPU-computed Sigma^2 grid.


def number_variance_gpu(
    unfolded: np.ndarray, lengths: np.ndarray | float, *, n_offsets: int = 2000
) -> np.ndarray:
    """GPU number variance Sigma^2(L). Must match the numpy reference in spacing.py."""
    cp = _cupy()
    x = cp.sort(cp.asarray(unfolded, dtype=cp.float64))
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    x0, xend = float(x[0]), float(x[-1])
    span = xend - x0
    out = np.full(lengths.shape, np.nan)
    for i, length in enumerate(lengths):
        if not (0.0 < length < span):
            continue
        starts = cp.linspace(x0, xend - length, n_offsets)
        counts = cp.searchsorted(x, starts + length, side="left") - cp.searchsorted(
            x, starts, side="left"
        )
        out[i] = float(counts.astype(cp.float64).var())
    return out


def dyson_mehta_delta3_gpu(
    unfolded: np.ndarray,
    lengths: np.ndarray | float,
    *,
    n_offsets: int = 2000,
    n_grid: int = 400,
) -> np.ndarray:
    """GPU Dyson-Mehta Delta_3(L) via the Mehta transform of a GPU Sigma^2 grid.

    The heavy part (Sigma^2 over the ``r`` grid) runs on the GPU; the cheap
    integral transform reuses ``spacing.delta3_from_sigma2`` on the host. Matches
    ``spacing.dyson_mehta_delta3`` to floating-point tolerance.
    """
    cp = _cupy()
    x = cp.sort(cp.asarray(unfolded, dtype=cp.float64))
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.float64))
    span = float(x[-1] - x[0])
    out = np.full(lengths.shape, np.nan)
    valid = (lengths > 0.0) & (lengths < span)
    if not valid.any():
        return out
    r = np.linspace(0.0, float(lengths[valid].max()), n_grid)
    s2 = np.empty_like(r)
    s2[0] = 0.0
    s2[1:] = number_variance_gpu(x, r[1:], n_offsets=n_offsets)
    out[valid] = spacing.delta3_from_sigma2(
        lambda q: np.interp(q, r, s2), lengths[valid], n_grid=n_grid
    )
    return out
