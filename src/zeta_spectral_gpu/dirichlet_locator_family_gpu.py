"""GPU batched-family Dirichlet-``L`` locator via CuPy + the family locator kernel.

The Phase-2 GPU "scale" leverage for #68: a Dirichlet family is embarrassingly
parallel -- each member's ``|M'_z(E)|`` scan is independent -- so the kernel
(``kernels/dirichlet_locator_family.cu``) evaluates the whole family across the
whole ``E``-grid in **one launch** (a ``(member, E-point)`` thread grid), instead
of looping member-by-member on the host (:func:`katz_sarnak.locate_member_zeros`).

This is the clean fp64 GPU win the #25/#51 forward rulings anticipated -- the
opposite regime from the flagship's mpmath-bound eigensolve: ``|M'_z|`` is
``O(1)``, cancellation-free, so double precision reproduces the CPU reference
(:func:`dirichlet_locator_family.family_scan`) to floating-point tolerance (the
house GPU-vs-CPU rule, pinned by ``tests/test_dirichlet_locator_family.py``).

Mirrors the single-character split (:mod:`dirac_mirror_gpu`) and the batched
``li_criterion_family`` split: :func:`dirichlet_locator_family.pack_family` does
the host prep (separable for a members/sec timing readout) and :func:`assemble_gpu`
runs the kernel. CuPy is imported lazily so the package imports on a CPU-only box.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import dirichlet_locator_family as fam
from .dirichlet_locator_family import PackedFamily

_KERNEL_SRC = Path(__file__).with_name("kernels") / "dirichlet_locator_family.cu"


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
    """Compile (once) and return the RawModule for dirichlet_locator_family.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


def assemble_gpu(
    packed: PackedFamily, grid: np.ndarray, *, block: int = 128
) -> np.ndarray:
    """Run the batched family locator kernel over ``grid`` for a packed family.

    Returns the ``(num_members, len(grid))`` ``complex128`` block of ``M'_z(E)`` --
    the same object as :func:`dirichlet_locator_family.family_scan`, computed in a
    single launch (one thread per ``(member, E-point)``). Dispatches to the real
    kernel (``family_mobius_locator``) for a real family and the complex kernel
    (``family_weighted_locator``) otherwise. The pure on-device assembly -- time
    this for a members/sec readout.
    """
    cp = _cupy()
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float64).ravel())
    n_e = int(grid.size)
    nc, n = packed.num_members, packed.n

    offsets_d = cp.asarray(packed.offsets, dtype=cp.int32)
    periods_d = cp.asarray(packed.periods, dtype=cp.int32)
    mu_amp_d = cp.asarray(packed.mu_amp, dtype=cp.float64)
    logk_d = cp.asarray(packed.logk, dtype=cp.float64)
    E_d = cp.asarray(grid)
    out_re = cp.empty(nc * n_e, dtype=cp.float64)
    out_im = cp.empty(nc * n_e, dtype=cp.float64)

    blocks = ((n_e + block - 1) // block, nc)
    if packed.is_real:
        chi_d = cp.asarray(packed.chi_re, dtype=cp.float64)
        kernel = _module().get_function("family_mobius_locator")
        args = (
            chi_d,
            offsets_d,
            periods_d,
            mu_amp_d,
            logk_d,
            np.int64(n),
            E_d,
            np.int64(n_e),
            np.int64(nc),
            out_re,
            out_im,
        )
    else:
        chi_re_d = cp.asarray(packed.chi_re, dtype=cp.float64)
        chi_im_d = cp.asarray(packed.chi_im, dtype=cp.float64)
        kernel = _module().get_function("family_weighted_locator")
        args = (
            chi_re_d,
            chi_im_d,
            offsets_d,
            periods_d,
            mu_amp_d,
            logk_d,
            np.int64(n),
            E_d,
            np.int64(n_e),
            np.int64(nc),
            out_re,
            out_im,
        )
    kernel(blocks, (block,), args)

    out = cp.asnumpy(out_re).reshape(nc, n_e) + 1j * cp.asnumpy(out_im).reshape(nc, n_e)
    return out


def family_scan_gpu(
    characters: list[np.ndarray],
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
    block: int = 128,
) -> np.ndarray:
    """Convenience: pack the family (host) + run the batched kernel (GPU).

    Returns the ``(num_members, len(grid))`` ``complex128`` ``M'_z`` block, which
    must reproduce :func:`dirichlet_locator_family.family_scan` on the same inputs
    (the house GPU-vs-CPU rule).
    """
    packed = fam.pack_family(characters, n, sigma=sigma)
    return assemble_gpu(packed, grid, block=block)
