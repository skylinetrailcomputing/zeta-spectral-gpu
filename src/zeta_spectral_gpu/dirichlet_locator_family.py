"""Batched-family Dirichlet-``L`` locator: packing + CPU reference (issue #68).

Phase-2 of the Katz-Sarnak family work (#51): where
:func:`katz_sarnak.locate_member_zeros` scans **one** member's ``|M'_z(E)|`` at a
time, the family is *embarrassingly parallel* -- hundreds of independent, cheap
fp64 scans. This module packs a whole family for the batched GPU kernel
(:mod:`dirichlet_locator_family_gpu`, ``kernels/dirichlet_locator_family.cu``) and
provides the **CPU reference** the kernel must reproduce (the house GPU-vs-CPU
rule).

The forward object per member is unchanged (Sierra arXiv:1404.4252 eq. 13.6): the
truncated partial sum of ``1 / L(s, chi)``,

    M'_z(E) = sum_{k<=n} chi(k) mu(k) k^{-(1/2 + iE)},

whose ``|M'_z(E)|`` peaks at the ordinates of the zeros of ``L(s, chi)``. The
character ``chi`` (its modulus is the only number-theoretic input) goes in and the
zeros come out -- no zero is ever consumed (forward, not inverse; see
:mod:`dirichlet` and the issue #25 ruling).

Pieces:

* :class:`PackedFamily` / :func:`pack_family` -- the family's characters packed for
  the kernel: a concatenated period-``q`` character table with per-member
  offsets/periods, plus the shared ``mu(k) k^{-sigma}`` and ``log k`` arrays. The
  per-member ``mu``/``log`` factors are shared, so only the small character table
  varies across the family.
* :func:`family_scan` -- the CPU reference: loops the established single-character
  :func:`dirac_mirror.mobius_partial_sum` over the family, returning the
  ``(num_members, len(grid))`` complex ``M'_z`` block. This is the ground truth the
  batched kernel is checked against.
* :func:`locate_family_peaks` -- turn a precomputed ``|M'_z|`` block into per-member
  located peaks (the forward output), reusing the single-character peak logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dirac_mirror import mobius_partial_sum, mobius_sieve
from .dirichlet import is_real_character, lfunction_weights
from .dirichlet_locator import local_maxima, peak_threshold


@dataclass(frozen=True)
class PackedFamily:
    """A Dirichlet family packed for the batched locator kernel.

    ``chi_re`` (and ``chi_im`` for a complex family) is the concatenation of every
    member's period-``q`` character table; member ``m`` occupies
    ``chi_re[offsets[m] : offsets[m] + periods[m]]`` and the kernel reads
    ``chi_re[offsets[m] + (k % periods[m])]`` for term ``k``. ``mu_amp[k-1] =
    mu(k) k^{-sigma}`` and ``logk[k-1] = log k`` are shared across the family
    (``k = 1..n``). ``is_real`` selects the fast real kernel
    (``family_mobius_locator``) over the complex one (``family_weighted_locator``).
    """

    chi_re: np.ndarray  # packed real character tables (float64)
    chi_im: np.ndarray | None  # packed imag tables, or None for a real family
    offsets: np.ndarray  # [num_members] int32 start index per member
    periods: np.ndarray  # [num_members] int32 period q_m
    mu_amp: np.ndarray  # [n] mu(k) k^{-sigma}, k=1..n (float64)
    logk: np.ndarray  # [n] log k, k=1..n (float64)
    is_real: bool
    num_members: int
    n: int
    sigma: float


def pack_family(
    characters: list[np.ndarray], n: int, *, sigma: float = 0.5
) -> PackedFamily:
    """Pack a family's characters + shared term arrays for the batched kernel.

    ``characters`` is a list of period-``q`` character tables (as
    :func:`dirichlet.dirichlet_character` / :func:`katz_sarnak.quadratic_character`
    return). The family is real iff **every** member is real; a single complex
    member upgrades the whole pack to the complex layout (real members keep their
    zero imaginary part). The shared ``mu(k) k^{-sigma}`` / ``log k`` arrays are
    built once. Separated from the scan so a caller can time the GPU assembly
    (members/sec) apart from this host prep.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    chars = [np.asarray(c, dtype=np.complex128) for c in characters]
    is_real = all(is_real_character(c) for c in chars)

    periods = np.array([c.size for c in chars], dtype=np.int32)
    offsets = np.zeros(len(chars), dtype=np.int32)
    if len(chars) > 1:
        offsets[1:] = np.cumsum(periods[:-1])
    table = np.concatenate(chars) if chars else np.empty(0, dtype=np.complex128)
    chi_re = np.ascontiguousarray(table.real, dtype=np.float64)
    chi_im = None if is_real else np.ascontiguousarray(table.imag, dtype=np.float64)

    mu = mobius_sieve(n)
    k = np.arange(1, n + 1, dtype=np.float64)
    mu_amp = np.ascontiguousarray(mu[1 : n + 1].astype(np.float64) * k**-sigma)
    logk = np.ascontiguousarray(np.log(k))

    return PackedFamily(
        chi_re=chi_re,
        chi_im=chi_im,
        offsets=offsets,
        periods=periods,
        mu_amp=mu_amp,
        logk=logk,
        is_real=is_real,
        num_members=len(chars),
        n=n,
        sigma=float(sigma),
    )


def family_scan(
    characters: list[np.ndarray],
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
) -> np.ndarray:
    """CPU reference batched scan: ``M'_z(E)`` for every member over ``grid``.

    Loops the established single-character locator
    :func:`dirac_mirror.mobius_partial_sum` (with
    :func:`dirichlet.lfunction_weights` for ``chi(k) mu(k)``) over the family,
    stacking the results into a ``(num_members, len(grid))`` ``complex128`` block.
    This is the ground truth the batched kernel
    (:func:`dirichlet_locator_family_gpu.assemble_gpu`) must reproduce to
    floating-point tolerance. No zero enters -- they are read *off* ``|M'_z|``.
    """
    grid = np.asarray(grid, dtype=np.float64)
    out = np.empty((len(characters), grid.size), dtype=np.complex128)
    for m, char in enumerate(characters):
        w = lfunction_weights(np.asarray(char), n)
        out[m] = mobius_partial_sum(grid, n, sigma=sigma, weights=w)
    return out


def locate_family_peaks(
    abs_m: np.ndarray,
    grid: np.ndarray,
    n: int,
    *,
    height: float | None = None,
) -> list[np.ndarray]:
    """Per-member located peaks from a precomputed ``|M'_z|`` block (forward output).

    ``abs_m`` is the ``(num_members, len(grid))`` magnitude block from a CPU or GPU
    scan. Returns one peak-ordinate array per member, using the same
    :func:`dirichlet_locator.peak_threshold` / :func:`dirichlet_locator.local_maxima`
    logic as the single-character locator. The peaks are the forward zeros; the
    independent mpmath zeros are used only to score them downstream.
    """
    abs_m = np.asarray(abs_m, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    h = peak_threshold(n) if height is None else height
    return [grid[local_maxima(row, h)] for row in abs_m]
