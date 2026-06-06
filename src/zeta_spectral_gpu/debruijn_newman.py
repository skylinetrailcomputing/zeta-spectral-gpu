"""De Bruijn–Newman forward flow — the issue #20 spike.

Forward ``H_t``-zero generator. We evaluate the entire function

    H_t(z) = integral_0^inf  e^{t u^2} Phi(u) cos(z u) du,
    Phi(u) = sum_{n>=1} (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) exp(-pi n^2 e^{4u}),

(Polymath15, arXiv:1904.12438) *directly from the structural kernel* ``Phi`` and
root-find its real zeros. At ``t = 0`` this is ``H_0(z) = xi(1/2 + i z/2) / 8``,
so the zeros of ``H_0`` sit at ``z = 2 gamma`` for each ordinate ``gamma`` of a
nontrivial zeta zero. The zeros are an *output* we characterise — we never feed
the known zeta zeros in (forward, not inverse; see README and project-framing).

The De Bruijn–Newman constant ``Lambda`` is defined by: ``H_t`` has only real
zeros iff ``t >= Lambda``. Rodgers–Tao give ``Lambda >= 0``, Polymath15 gives
``Lambda <= 0.22``, and RH <=> ``Lambda <= 0``. For ``t >= 0`` the flow keeps the
zeros real and makes them progressively *more rigid* (toward an evenly spaced
"picket fence" as ``t -> inf``). That monotone increase in rigidity is exactly
what this spike measures by feeding the generated zeros through the #15
statistics (``spacing.number_variance`` / ``dyson_mehta_delta3``).

Precision, not throughput, is the constraint here. ``H_t`` at height ``z`` is
exponentially small — ``|H_0(2 gamma)| ~ exp(-pi gamma / 4)`` — relative to the
O(1) integrand, so the oscillatory integral cancels catastrophically and must run
in extended precision. That is why this is a CPU/``mpmath`` spike and the GPU does
not feature: the bottleneck is digits, not parallel work (see the precision note
in CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import mpmath as mp
import numpy as np

from .zeros import smooth_count

DATA = Path(__file__).resolve().parents[2] / "data"
LN10 = float(np.log(10.0))


def phi(u, *, max_terms: int = 1000):
    """The de Bruijn kernel ``Phi(u)`` at the current ``mpmath`` precision.

    Sums the theta-derived series until the terms fall below the working epsilon.
    ``Phi(u) > 0`` for all real ``u`` (each bracket ``2 pi n^2 e^{4u} - 3`` is
    ``>= 2 pi - 3 > 0``), and it decays super-exponentially like
    ``exp(-pi e^{4u})``, so the cosine transform converges as an entire function.
    """
    u = mp.mpf(u)
    e4 = mp.exp(4 * u)
    e5 = mp.exp(5 * u)
    e9 = mp.exp(9 * u)
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    total = mp.mpf(0)
    for n in range(1, max_terms + 1):
        n2 = mp.mpf(n) ** 2
        term = (2 * mp.pi**2 * n2**2 * e9 - 3 * mp.pi * n2 * e5) * mp.exp(
            -mp.pi * n2 * e4
        )
        total += term
        if n > 1 and abs(term) < eps * (abs(total) + 1):
            break
    return total


def _u_max(t, *, guard: int = 12) -> float:
    """Upper integration limit where the integrand drops below the precision.

    ``Phi`` decays like ``exp(-pi e^{4u})``; we solve ``pi e^{4u} = N`` for ``N``
    a comfortable margin past the working precision, the ``+30`` covering the
    ``e^{9u} e^{t u^2}`` prefactor growth.
    """
    target = (mp.mp.dps + guard) * LN10 + 30.0
    return float(np.log(target / np.pi) / 4.0) + 0.05


def h_t(z, t, *, u_max=None):
    """``H_t(z)`` for real ``z``, ``t`` at the current ``mpmath`` precision (mpf).

    The ``cos(z u)`` oscillation is handled by subdividing ``[0, u_max]`` at the
    half-periods ``pi / z`` and applying Gauss–Legendre on each (smooth) panel —
    the robust pattern for an oscillatory integrand with a known frequency. (GL
    matches adaptive tanh-sinh to full precision here but is ~4x faster, because
    each half-period panel is well approximated by a modest-degree polynomial; the
    exponentially small value at height still resolves since the per-panel rule
    reaches the working precision.)
    """
    z = mp.mpf(z)
    t = mp.mpf(t)
    if u_max is None:
        u_max = _u_max(t)
    u_max = mp.mpf(u_max)

    def f(u):
        return mp.exp(t * u * u) * phi(u) * mp.cos(z * u)

    if z == 0:
        return mp.quadgl(f, [0, u_max])
    half = mp.pi / z
    n_half = int(mp.floor(u_max / half))
    points = [k * half for k in range(n_half + 1)]
    if points[-1] < u_max:
        points.append(u_max)
    return mp.quadgl(f, points)


def xi(s):
    """Riemann ``xi(s) = (s(s-1)/2) pi^{-s/2} Gamma(s/2) zeta(s)`` (mpmath)."""
    return (s * (s - 1) / 2) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def h0_reference(z):
    """``H_0(z)`` via the closed form ``xi(1/2 + i z/2) / 8`` — the integral target.

    Real for real ``z``. Used only to validate the forward integral generator and
    to pin the ``z = 2 gamma`` normalisation; it is *not* the forward generator.
    """
    s = mp.mpc(mp.mpf(1) / 2, mp.mpf(z) / 2)
    return (xi(s) / 8).real


def _ordinate_for_index(n: int) -> float:
    """Approximate height ``gamma`` of the ``n``-th ordinate: invert ``N_bar = n``."""
    lo, hi = 10.0, 30.0
    while smooth_count(hi) < n:
        hi *= 1.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if smooth_count(mid) < n:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def required_dps(n: int, *, guard: int = 18) -> int:
    """Working precision so ``H_0`` stays resolvable up to the ``n``-th zero.

    ``|H_0(2 gamma)| ~ exp(-pi gamma / 4)``, so at the top height ``gamma_n`` the
    integral loses ``pi gamma_n / (4 ln 10)`` digits to cancellation; we keep
    ``guard`` digits beyond that.
    """
    gamma_max = _ordinate_for_index(n)
    lost = np.pi * gamma_max / 4.0 / LN10
    return int(np.ceil(lost)) + guard


def h_t_zeros(
    t,
    n: int,
    *,
    dps: int | None = None,
    z_lo: float = 1.0,
    scan_step: float = 0.5,
    z_pad: float = 1.25,
    cache_dir: Path | None = DATA,
    refresh: bool = False,
) -> np.ndarray:
    """First ``n`` positive real zeros of ``H_t``, forward-generated from ``Phi``.

    Scans ``H_t`` on a grid of step ``scan_step`` (at the precision ``dps`` needed
    to get the *sign* right at height), brackets each sign change, and refines it.
    Results are cached to CSV keyed by ``(t, n, dps)`` so reruns are instant.

    For ``t >= 0`` and the heights probed here (where RH holds, so ``Lambda <= 0``)
    every zero of ``H_t`` is real, so a real-axis scan finds them all.
    """
    if dps is None:
        dps = required_dps(n)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"dbn_zeros_t{float(t):g}_n{n}_dps{dps}.csv"
        if cache.exists() and not refresh:
            vals = np.atleast_1d(np.loadtxt(cache))
            if vals.size >= n:
                return vals[:n]

    gamma_max = _ordinate_for_index(n)
    z_max = 2.0 * gamma_max * z_pad
    roots: list[float] = []
    with mp.workdps(dps):
        u_max = _u_max(t)

        def f(z):
            return h_t(z, t, u_max=u_max)

        step = mp.mpf(scan_step)
        z0 = mp.mpf(z_lo)
        f0 = f(z0)
        while z0 < z_max and len(roots) < n:
            z1 = z0 + step
            f1 = f(z1)
            if f0 == 0:
                roots.append(float(z0))
            elif f0 * f1 < 0:
                root = mp.findroot(f, (z0, z1), solver="anderson")
                roots.append(float(root))
            z0, f0 = z1, f1

    out = np.asarray(roots[:n], dtype=np.float64)
    if cache is not None and out.size:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(cache, out)
    return out


def empirical_unfold(z: np.ndarray, *, degree: int = 6) -> np.ndarray:
    """Unfold real levels ``z`` by their *own* smooth counting function.

    Fits a degree-``degree`` polynomial to the staircase ``(z_j, j - 1/2)`` and
    returns ``w_j = poly(z_j)``, so ``np.diff(w)`` has unit mean. This is
    density-agnostic, which is the point: the heat flow changes the density of the
    ``H_t`` zeros, so unfolding each set by its own count (rather than the fixed
    ``t = 0`` Riemann–von Mangoldt ``N_bar``) isolates *rigidity* from a trivial
    density shift, and applies fairly across different ``t``.
    """
    z = np.sort(np.asarray(z, dtype=np.float64))
    counts = np.arange(z.size, dtype=np.float64) + 0.5
    coeffs = np.polyfit(z, counts, degree)
    return np.polyval(coeffs, z)
