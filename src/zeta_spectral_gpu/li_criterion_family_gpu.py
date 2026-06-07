"""GPU family Li-coefficient assembly via CuPy + ``kernels/li_criterion_family.cu``.

The Phase-2 GPU "scale" leverage for #71: the family of Dirichlet characters is
embarrassingly parallel (each ``lambda_n(chi)`` is an independent, zero-free forward
computation), so the kernel maps **one CUDA block per character** and assembles the
Li coefficients for the whole family at once, in fp64.

The division of labour follows CLAUDE.md's precision rule. The precision-delicate
analytic inputs -- the Taylor coefficients of ``log L(1+u, chi)`` (the character's
generalized-Stieltjes data) -- are computed in ``mpmath`` on the host
(:func:`li_criterion_family.l_taylor_coefficients`) and downcast to fp64. The GPU
then does the fp64 *assembly*: the ``log``-power-series recurrence, the parity-aware
``Gamma``/``(q/pi)`` terms, the Bombieri-Lagarias combination, and the
``min Re lambda_n`` / GRH-positivity reduction. fp64 saturates the binomial
cancellation past a few dozen coefficients (the repo's recurring precision wall), so
the GPU path is the **small-``n`` family producer**; deeper ``n`` stays mpmath's job
(:mod:`li_criterion_family`), and ``tests/test_li_criterion_family.py`` pins
GPU-vs-CPU agreement on small ``n``.

This mirrors the Katz-Sarnak ``#51 -> #68`` split: a GPU generalized-Stieltjes kernel
that would also produce the analytic inputs on-device is the documented follow-up.
CuPy is imported lazily so the package imports on machines without a GPU.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np

from . import li_criterion as li
from . import li_criterion_family as fam

_KERNEL_SRC = Path(__file__).with_name("kernels") / "li_criterion_family.cu"


def _cupy():
    """Import cupy on demand, with a clear error if the GPU extra is missing."""
    from ._cuda_dll import add_cuda_dll_directories

    add_cuda_dll_directories()  # Windows: nvidia/*/bin on the DLL path
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
    """Compile (once) and return the RawModule for li_criterion_family.cu."""
    cp = _cupy()
    src = _KERNEL_SRC.read_text(encoding="utf-8")
    return cp.RawModule(code=src, options=("--std=c++14",))


@dataclass
class FamilyInputs:
    """fp64 host inputs to the family Li kernel (one row per character).

    ``d_re``/``d_im`` are the normalized ``L``-Taylor coefficients ``d_k = c_k/c_0``
    (``k = 1..n_max``) flattened ``[char * n_max + (k-1)]``; ``poly`` and ``binom`` are
    the parity-shared / index-only tables the kernel reuses across the family.
    """

    d_re: np.ndarray
    d_im: np.ndarray
    logqpi: np.ndarray
    parity: np.ndarray
    poly: np.ndarray
    binom: np.ndarray
    num_chars: int
    n_max: int


def prepare_inputs(
    characters: list[tuple[str, np.ndarray]], n_max: int, *, dps: int | None = None
) -> FamilyInputs:
    """Build the fp64 kernel inputs for a family (the mpmath/host prep step).

    Computes each character's ``L``-Taylor coefficients in ``mpmath``
    (:func:`li_criterion_family.l_taylor_coefficients`) and the parity-shared
    polygamma / binomial tables, downcasting to fp64. Separated from
    :func:`assemble_gpu` so a caller can time the GPU assembly (families/sec) apart
    from this precision-delicate prep.
    """
    if dps is None:
        dps = li.default_dps(n_max)
    num_chars = len(characters)
    d_re = np.zeros(num_chars * n_max, dtype=np.float64)
    d_im = np.zeros(num_chars * n_max, dtype=np.float64)
    logqpi = np.zeros(num_chars, dtype=np.float64)
    parity = np.zeros(num_chars, dtype=np.int32)

    with mp.workdps(dps):
        for c, (_, char) in enumerate(characters):
            char_arr = np.asarray(char)
            q = char_arr.size
            coeffs = fam.l_taylor_coefficients(char_arr, n_max, dps=dps)
            c0 = coeffs[0]
            for k in range(1, n_max + 1):
                d = coeffs[k] / c0
                d_re[c * n_max + (k - 1)] = float(mp.re(d))
                d_im[c * n_max + (k - 1)] = float(mp.im(d))
            logqpi[c] = float(mp.log(mp.mpf(q) / mp.pi))
            parity[c] = fam.character_parity(char_arr)

        poly = np.zeros(2 * n_max, dtype=np.float64)
        for a in (0, 1):
            arg = (1 + a) * (mp.mpf(1) / 2)
            for k in range(1, n_max + 1):
                poly[a * n_max + (k - 1)] = float(
                    mp.polygamma(k - 1, arg) / (mp.mpf(2) ** k * mp.factorial(k))
                )
        binom = np.zeros(n_max * n_max, dtype=np.float64)
        for n in range(1, n_max + 1):
            for j in range(n):
                binom[(n - 1) * n_max + j] = float(mp.binomial(n - 1, j))

    return FamilyInputs(
        d_re=d_re,
        d_im=d_im,
        logqpi=logqpi,
        parity=parity,
        poly=poly,
        binom=binom,
        num_chars=num_chars,
        n_max=n_max,
    )


def assemble_gpu(
    inputs: FamilyInputs, *, block: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the family Li kernel on prepared fp64 inputs (one block per character).

    Returns ``(lam, min_re, all_pos)``: ``lam`` is ``(num_chars, n_max)`` complex
    ``lambda_n(chi)``, ``min_re`` is the per-character ``min_n Re lambda_n``, and
    ``all_pos`` is the per-character GRH-positivity flag. The pure on-device
    assembly -- time this for a families/sec readout.
    """
    cp = _cupy()
    nc, N = inputs.num_chars, inputs.n_max

    d_re = cp.asarray(inputs.d_re, dtype=cp.float64)
    d_im = cp.asarray(inputs.d_im, dtype=cp.float64)
    logqpi = cp.asarray(inputs.logqpi, dtype=cp.float64)
    parity = cp.asarray(inputs.parity, dtype=cp.int32)
    poly = cp.asarray(inputs.poly, dtype=cp.float64)
    binom = cp.asarray(inputs.binom, dtype=cp.float64)

    lam_re = cp.empty(nc * N, dtype=cp.float64)
    lam_im = cp.empty(nc * N, dtype=cp.float64)
    min_re = cp.empty(nc, dtype=cp.float64)
    all_pos = cp.empty(nc, dtype=cp.int32)

    kernel = _module().get_function("family_li")
    shared = 2 * (N + 1) * 8  # are[N+1] + aim[N+1] doubles
    kernel(
        (nc,),
        (block,),
        (
            d_re,
            d_im,
            logqpi,
            parity,
            poly,
            binom,
            np.int32(N),
            np.int32(nc),
            lam_re,
            lam_im,
            min_re,
            all_pos,
        ),
        shared_mem=shared,
    )
    lam = cp.asnumpy(lam_re).reshape(nc, N) + 1j * cp.asnumpy(lam_im).reshape(nc, N)
    return lam, cp.asnumpy(min_re), cp.asnumpy(all_pos).astype(bool)


def family_li_coefficients_gpu(
    characters: list[tuple[str, np.ndarray]],
    n_max: int,
    *,
    dps: int | None = None,
    block: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: host prep (mpmath) + GPU assembly for a family.

    Returns ``(lam, min_re, all_pos)`` as in :func:`assemble_gpu`. The fp64 ``lam``
    must reproduce the mpmath reference
    (:func:`li_criterion_family.character_li_coefficients`) on small ``n`` -- the
    house GPU-vs-CPU rule.
    """
    inputs = prepare_inputs(characters, n_max, dps=dps)
    return assemble_gpu(inputs, block=block)
