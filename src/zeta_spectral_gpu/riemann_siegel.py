"""Riemann-Siegel evaluator for the Hardy ``Z`` function and ``zeta`` on the
critical line (issue #55) -- the CPU reference.

The Riemann-Siegel formula computes ``zeta(1/2 + i t)`` from a short main sum plus
an asymptotic remainder, costing ``O(sqrt(t))`` instead of the
``O(t)`` of a naive Dirichlet sum or the slow convergence of the Euler-Maclaurin
path. With ``tau = sqrt(t / 2pi)`` and ``N = floor(tau)``,

    Z(t) = 2 sum_{n=1}^{N} cos(theta(t) - t log n) / sqrt(n)
           + (-1)^{N-1} tau^{-1/2} [ C_0(p) + C_1(p) tau^{-1} + C_2(p) tau^{-2} ],

where ``theta`` is the Riemann-Siegel theta function, ``p = tau - N`` is the
fractional part, and ``Z(t) = e^{i theta(t)} zeta(1/2 + i t)`` is the (real-valued)
Hardy function, so ``zeta(1/2 + i t) = Z(t) e^{-i theta(t)}`` and the zeros of
``zeta`` on the line are the real zeros of ``Z``.

The correction coefficients ``C_k(p)`` are built from the single function

    Psi(p) = cos(2pi (p^2 - p - 1/16)) / cos(2pi p)

and its derivatives (Gabcke / Edwards Ch. 7):

    C_0 = Psi,
    C_1 = -Psi'''  / (96 pi^2),
    C_2 =  Psi^(6) / (18432 pi^4) + Psi'' / (64 pi^2).

These three terms reproduce ``mpmath.siegelz`` to ~1e-6 by ``t = 50`` and
~1e-11 by ``t = 1e5`` (the expansion is asymptotic: more accurate the higher the
height). ``Psi`` is *entire* -- the apparent poles of the quotient at
``p in {1/4, 3/4}`` are removable -- and its derivatives are evaluated in fp64 by
a Cauchy integral on a complex contour that dodges those points
(:func:`_psi_derivatives`), stable everywhere and with no ``mpmath`` in the hot
path -- which is what lets the main sum run as a plain fp64 GPU reduction
(:mod:`riemann_siegel_gpu`).

Precision regime: like :mod:`dirac_mirror`, and unlike the flagship CCM eigensolve,
this lives in an ``O(1)``-magnitude, cancellation-free world -- the main sum is
``O(sqrt(t))`` terms of size ``O(1)`` adding to an ``O(1)`` value -- so fp64 is
sufficient at the heights of interest and the GPU port is a clean win (contrast
:mod:`debruijn_newman`, whose ``H_t`` integral *is* precision-bound and stays on
``mpmath``).

Forward, not inverse: this is a *tool*, not an experiment. It evaluates ``zeta``
from its Riemann-Siegel expansion -- no zeros are consumed. Anything built on top
of it (zero verification for the locator / Katz-Sarnak family work, value
distribution, De Bruijn-Newman at height) must keep the zeros an output. A test
cross-checks every piece against ``mpmath`` (``siegeltheta`` / ``siegelz`` /
``zeta``), the house ground-truth rule.
"""

from __future__ import annotations

from math import factorial

import numpy as np

TWO_PI = 2.0 * np.pi

# Removable singularities of Psi (and their nearest periodic images) on the real
# axis: cos(2pi p) and the numerator vanish together at p = 1/4, 3/4 (mod 1).
_PSI_SINGULARITIES = np.array([-0.25, 0.25, 0.75, 1.25])
# Candidate contour radii for the Cauchy-integral derivatives; one is chosen per
# point so the circle's two real-axis crossings dodge _PSI_SINGULARITIES.
_CONTOUR_RADII = np.array([0.30, 0.36, 0.42])
_CONTOUR_NODES = 48


def theta(t: np.ndarray | float) -> np.ndarray | float:
    """Riemann-Siegel theta ``theta(t)`` via its asymptotic series (fp64).

    ``theta(t) = arg Gamma(1/4 + i t/2) - (t/2) log pi``; the standard asymptotic

        theta(t) ~ (t/2) log(t/2pi) - t/2 - pi/8
                   + 1/(48 t) + 7/(5760 t^3) + 31/(80640 t^5) + 127/(430080 t^7)

    matches ``mpmath.siegeltheta`` to <3e-13 by ``t = 200`` (a test pins this), so
    it is used directly on both the CPU and GPU paths -- keeping ``Z`` consistent
    between them to floating-point tolerance. Valid for ``t`` away from the small
    transition region; intended for ``t >~ 15``.
    """
    t = np.asarray(t, dtype=np.float64)
    inv = 1.0 / t
    inv2 = inv * inv
    series = inv * (
        1.0 / 48.0
        + inv2 * (7.0 / 5760.0 + inv2 * (31.0 / 80640.0 + inv2 * (127.0 / 430080.0)))
    )
    out = (t / 2.0) * np.log(t / TWO_PI) - t / 2.0 - np.pi / 8.0 + series
    return out if out.ndim else float(out)


def _contour_radii(p: np.ndarray) -> np.ndarray:
    """Per-point Cauchy-contour radius dodging Psi's removable singularities.

    For each ``p`` pick the candidate radius (``_CONTOUR_RADII``) whose circle's
    two real-axis crossings ``p +/- R`` sit furthest from ``_PSI_SINGULARITIES`` --
    the only places where evaluating ``Psi`` on the contour would cancel in fp64.
    (The *disk* may contain a singularity; that is harmless because ``Psi`` is
    entire -- the apparent poles are removable.)
    """
    crossings = np.stack(
        [p[:, None] - _CONTOUR_RADII, p[:, None] + _CONTOUR_RADII], axis=2
    )  # (n, n_radii, 2)
    dist = np.abs(crossings[..., None] - _PSI_SINGULARITIES).min(axis=-1)
    return _CONTOUR_RADII[np.argmax(dist.min(axis=2), axis=1)]


def _psi_derivatives(p: np.ndarray, order: int) -> list[np.ndarray]:
    """``Psi(p), Psi'(p), ..., Psi^(order)(p)`` in fp64, vectorized over ``p``.

    ``Psi(p) = cos(2pi(p^2 - p - 1/16)) / cos(2pi p)`` is *entire* (numerator and
    denominator share simple zeros at ``p in {1/4, 3/4} mod 1``, so those poles are
    removable). The straightforward quotient/recurrence for its derivatives divides
    by ``cos(2pi p)`` once per order, which is catastrophic near those points -- so
    instead each derivative is read off a single Cauchy integral on a circle in the
    complex plane,

        Psi^(k)(p) = k! / (2pi i) oint Psi(z) / (z - p)^{k+1} dz
                   ~ k! / (M R^k) sum_{j} Psi(p + R e^{i theta_j}) e^{-i k theta_j},

    a trapezoidal rule over ``M = _CONTOUR_NODES`` equally spaced nodes. The radius
    ``R`` (:func:`_contour_radii`) keeps the contour away from the removable points,
    where ``Psi`` is then ``O(1)`` everywhere on the circle; the rule is spectrally
    accurate for the entire ``Psi``, giving every derivative 0..order from one set
    of evaluations and matching ``mpmath`` to ~1e-10 through order 6 (even *at*
    ``p = 1/4``). Returns the real parts (``Psi`` is real on the real axis).
    """
    p = np.asarray(p, dtype=np.float64)
    shape = p.shape
    pf = p.ravel()
    radius = _contour_radii(pf)  # (n,)
    theta = TWO_PI * np.arange(_CONTOUR_NODES) / _CONTOUR_NODES  # (M,)
    z = pf[:, None] + radius[:, None] * np.exp(1j * theta)[None, :]  # (n, M)
    psi_z = np.cos(TWO_PI * (z * z - z - 1.0 / 16.0)) / np.cos(TWO_PI * z)
    out = []
    for k in range(order + 1):
        weighted = psi_z * np.exp(-1j * k * theta)[None, :]
        deriv = factorial(k) / (_CONTOUR_NODES * radius**k) * np.real(weighted.sum(1))
        out.append(deriv.reshape(shape))
    return out


# correction order needed from Psi: C_0 -> Psi, C_1 -> Psi''', C_2 -> Psi^(6)
_REMAINDER_ORDER = {0: 0, 1: 3, 2: 6}


def rs_remainder(
    t: np.ndarray | float, *, correction_terms: int = 2
) -> np.ndarray | float:
    """Riemann-Siegel remainder ``(-1)^{N-1} tau^{-1/2} sum_k C_k(p) tau^{-k}``.

    The additive correction past the main sum, fp64 and vectorized over ``t``.
    ``correction_terms`` selects how many ``C_k`` to include (0, 1, or 2); each
    extra term sharpens the asymptotic accuracy (see the module docstring). Pure
    fp64 -- this is what the GPU path adds on the host after the kernel returns the
    main sum.
    """
    if correction_terms not in _REMAINDER_ORDER:
        raise ValueError("correction_terms must be 0, 1, or 2")
    t = np.asarray(t, dtype=np.float64)
    tau = np.sqrt(t / TWO_PI)
    n = np.floor(tau).astype(np.int64)
    p = tau - n
    inv_tau = 1.0 / tau  # = (t/2pi)^{-1/2}; term k carries inv_tau^k, prefactor ^{1/2}

    d = _psi_derivatives(p, _REMAINDER_ORDER[correction_terms])
    coeffs = [d[0]]
    if correction_terms >= 1:
        coeffs.append(-d[3] / (96.0 * np.pi**2))
    if correction_terms >= 2:
        coeffs.append(d[6] / (18432.0 * np.pi**4) + d[2] / (64.0 * np.pi**2))

    bracket = np.zeros_like(t)
    for k, c in enumerate(coeffs):
        bracket += c * inv_tau**k
    sign = np.where(n % 2 == 1, 1.0, -1.0)  # (-1)^{N-1}
    out = sign * np.sqrt(inv_tau) * bracket
    return out if out.ndim else float(out)


def main_sum(t: np.ndarray | float, *, theta_t: np.ndarray | None = None) -> np.ndarray:
    """Riemann-Siegel main sum ``2 sum_{n<=N} cos(theta - t log n)/sqrt(n)`` (fp64).

    The ``O(sqrt(t))`` dominant term, vectorized over ``t``. Each height has its own
    ``N = floor(sqrt(t/2pi))``; the loop runs to the grid-wide maximum and masks the
    terms with ``n > N(t)`` -- the direct CPU analogue of the per-thread kernel in
    :mod:`riemann_siegel_gpu` (which loops to its own ``N`` instead). ``theta_t`` may
    be supplied to avoid recomputing :func:`theta`.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    th = theta(t) if theta_t is None else np.asarray(theta_t, dtype=np.float64)
    th = np.atleast_1d(th)
    big_n = np.floor(np.sqrt(t / TWO_PI)).astype(np.int64)
    n_max = int(big_n.max()) if big_n.size else 0
    acc = np.zeros_like(t)
    for n in range(1, n_max + 1):
        mask = n <= big_n
        acc[mask] += np.cos(th[mask] - t[mask] * np.log(n)) / np.sqrt(n)
    return 2.0 * acc


def hardy_z(t: np.ndarray | float, *, correction_terms: int = 2) -> np.ndarray | float:
    """Hardy ``Z(t)`` via Riemann-Siegel -- real-valued, fp64.

    ``Z(t) = e^{i theta(t)} zeta(1/2 + i t)`` is real for real ``t``; its real
    zeros are exactly the ordinates of the zeta zeros on the critical line, so a
    sign change of ``Z`` brackets a zero (the fast, precision-friendly way to
    *verify* located zeros -- the zeros stay an output). Reproduces
    ``mpmath.siegelz`` to the asymptotic accuracy set by ``correction_terms``.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    z = main_sum(t_arr) + np.atleast_1d(
        rs_remainder(t_arr, correction_terms=correction_terms)
    )
    if t_arr.ndim == 0:
        return float(z[0])
    return z.reshape(t_arr.shape)


def zeta_critical(
    t: np.ndarray | float, *, correction_terms: int = 2
) -> np.ndarray | complex:
    """``zeta(1/2 + i t) = Z(t) e^{-i theta(t)}`` via Riemann-Siegel (fp64 complex).

    The on-line zeta value itself (magnitude and phase), recovered from the real
    Hardy function and the theta phase. Matches ``mpmath.zeta(1/2 + i t)`` to the
    same asymptotic accuracy as :func:`hardy_z`.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    z = hardy_z(t_arr, correction_terms=correction_terms)
    out = np.atleast_1d(z) * np.exp(-1j * np.atleast_1d(theta(t_arr)))
    if t_arr.ndim == 0:
        return complex(out[0])
    return out.reshape(t_arr.shape)
