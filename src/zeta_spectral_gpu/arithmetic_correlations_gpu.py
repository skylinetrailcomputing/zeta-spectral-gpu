"""GPU zero Fourier statistic via the kernel in ``kernels/arithmetic_correlations.cu``.

Mirrors ``arithmetic_correlations.zero_fourier`` exactly (the house GPU-vs-CPU
rule); it exists because the empirical form factor is an O(N * M) oscillatory
sum — embarrassingly parallel over the frequency grid, which is where the
RTX 3090 earns its keep at 10^6-zero scale. The empirical R2 histogram needs
no new kernel: ``spacing_gpu.pair_correlation_histogram_gpu`` already bins raw
separations.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import arithmetic_correlations as ac
from .spacing_gpu import _cupy

_KERNEL_SRC = Path(__file__).with_name("kernels") / "arithmetic_correlations.cu"
_BLOCK = 256  # must match #define BLOCK in the kernel source


@functools.lru_cache(maxsize=1)
def _module():
    """Compile (once) and return the RawModule for arithmetic_correlations.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


def zero_fourier_gpu(
    tau: np.ndarray,
    u: np.ndarray | float,
    t_lo: float,
    t_hi: float,
    *,
    window: str = "hann",
) -> np.ndarray:
    """GPU ``S(u) = sum_n w(tau_n) e^{i u tau_n}``. Must match the CPU reference.

    Window weights are formed on the host (cheap, one pass) so the taper logic
    has a single owner in ``arithmetic_correlations.window_weights``.
    """
    cp = _cupy()
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    tau = np.asarray(tau, dtype=np.float64)
    w = ac.window_weights(tau, t_lo, t_hi, window)
    keep = w > 0.0
    tau_d = cp.asarray(tau[keep])
    w_d = cp.asarray(w[keep])
    u_d = cp.asarray(u)
    out_re = cp.empty(u.size, dtype=cp.float64)
    out_im = cp.empty(u.size, dtype=cp.float64)

    kernel = _module().get_function("zero_fourier")
    kernel(
        (u.size,),
        (_BLOCK,),
        (
            tau_d,
            w_d,
            np.int64(tau_d.size),
            u_d,
            np.int64(u.size),
            out_re,
            out_im,
        ),
    )
    return cp.asnumpy(out_re) + 1j * cp.asnumpy(out_im)
