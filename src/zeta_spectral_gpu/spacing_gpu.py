"""GPU statistics via CuPy + the hand-written kernels in ``kernels/spacing.cu``.

These mirror ``spacing.py`` exactly; they exist to run the same reductions at
large N. CuPy is imported lazily so the package imports on machines without a
GPU (and without the optional ``gpu`` extra installed).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

_KERNEL_SRC = Path(__file__).with_name("kernels") / "spacing.cu"


def _cupy():
    """Import cupy on demand, with a clear error if the GPU extra is missing."""
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
        (x, np.int64(n), np.float64(bin_width), np.int64(n_bins),
         np.float64(max_sep), hist),
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
