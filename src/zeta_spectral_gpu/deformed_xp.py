"""Sierra's deformed-``xp`` operator — a forward candidate spectrum (issue #23).

Germán Sierra & Javier Rodríguez-Laguna, *The H=xp model revisited and the
Riemann zeros*, PRL **106**, 200201 (2011), arXiv:1102.5356.

Berry and Keating's ``H = xp`` reproduces the *average* Riemann zeros
semiclassically but has open (non-periodic) classical orbits. Sierra &
Rodríguez-Laguna close the orbits with a geometric deformation,

    H = x (p + l_p^2 / p),      x >= l_x,   p in R                      (eq. 3)

whose normal-ordered quantization is the self-adjoint operator

    H_hat = x^{1/2} (p_hat + l_p^2 / p_hat) x^{1/2}                     (eq. 8)

with ``1/p_hat`` the 1D Green's function ``<x|1/p_hat|y> = -(i/hbar) theta(y-x)``
(eq. 9). ``H_hat`` is therefore *integro-differential* — it carries a nonlocal
integral term (eq. 10) — and is self-adjoint on a U(1) family of extensions
parameterised by an angle ``theta`` (deficiency indices (1,1), so every
eigenvalue is real), fixed by the nonlocal boundary condition (eq. 11). Its
eigenfunctions are ``psi_E(x) = x^{iE/2} K_{1/2 - iE/2}(l_p x)`` (eq. 12) and the
eigenenergies are the real roots of the secular equation (eq. 14)

    Xi(E) = e^{-i theta/2} K_{1/2 + iE/2}(h/hbar)
          + e^{+i theta/2} K_{1/2 - iE/2}(h/hbar) = 0,     h = l_x l_p.

We use units ``hbar = 1`` and ``l_x l_p = 2*pi`` (the Bessel argument is ``2*pi``;
only the product enters, by the model's scaling symmetry), and take
``theta = pi/4`` — the Riemann self-adjoint extension (eq. 24, q=1; equivalently
the rule that matches ``N(E)`` to half-integers ``n + 1/2``). For real ``E``,
``K_{1/2 - iE/2}(x)`` is the complex conjugate of ``K_{1/2 + iE/2}(x)``, so

    Xi(E) = 2 * Re[ e^{-i theta/2} K_{1/2 + iE/2}(2*pi) ]

is real-valued and its real zeros are the spectrum. (Note: arXiv:1102.5356's
eq. 14 is used verbatim here; a paraphrase that writes the two K-terms with a
relative minus sign corresponds to ``theta -> theta + pi``.)

Forward, not inverse (see ``README`` / ``knowledge/project-framing.md``): the
operator is a *geometric* deformation of ``xp`` — no primes and no zeros are
consumed. Its spectrum reproduces the **average** Riemann zeros (the smooth
Riemann–von Mangoldt count ``zeros.smooth_count``) but **none** of the GUE
fluctuations; Sierra notes the fluctuations are absent / an open problem. That is
the point of the warm-up: matching the mean density is necessary but nowhere near
sufficient — the bar the flagship must clear at the *fluctuation* level. The zeros
appear only downstream as an output comparison; the universality statistics on top
of this spectrum live in the companion issue (#24).

This module is the CPU secular reference (``mpmath`` root-finding), mirroring the
``#5/#6/#15`` reference-first structure. The GPU dense-matrix assembly of the
integro-differential operator (eq. 10) -> ``cupy.linalg.eigh`` is a separate
follow-up: the first-order nonlocal operator needs a decaying (Galerkin) basis to
avoid spurious particle-in-a-box modes, which is its own numerical-methods task.
"""

from __future__ import annotations

from pathlib import Path

import mpmath as mp
import numpy as np

from .zeros import smooth_count

DATA = Path(__file__).resolve().parents[2] / "data"
LN10 = float(np.log(10.0))
TWO_PI = 2.0 * np.pi

# eq. 24 with q = 1: the self-adjoint extension that reproduces the average zeros.
THETA_RIEMANN = float(np.pi / 4.0)
# Bessel argument h / hbar with hbar = 1 and h = l_x l_p = 2*pi.
BESSEL_ARG = TWO_PI
# Classical energy floor |E| >= 2h = 4*pi (eq. 6); no eigenvalues below it.
CLASSICAL_BOUND = 2.0 * TWO_PI


def _secular_full(E, theta: float = THETA_RIEMANN, *, arg: float = BESSEL_ARG):
    """The full complex secular form (eq. 14) as an ``mpc``.

    Evaluates both ``K`` orders independently; for real ``E`` the imaginary part
    is zero (the operator is self-adjoint), which ``secular`` exploits. Kept
    separate so a test can confirm that reality rather than assume it.
    """
    nu = mp.mpc(mp.mpf(1) / 2, mp.mpf(E) / 2)  # 1/2 + iE/2
    kp = mp.besselk(nu, arg)
    km = mp.besselk(mp.conj(nu), arg)  # 1/2 - iE/2 = conjugate order
    half = mp.mpf(theta) / 2
    return mp.e ** (mp.mpc(0, -half)) * kp + mp.e ** (mp.mpc(0, half)) * km


def secular(E, theta: float = THETA_RIEMANN, *, arg: float = BESSEL_ARG):
    """Real-valued secular function ``Xi(E)`` (eq. 14); zero at an eigenenergy.

    ``Xi(E) = 2 Re[e^{-i theta/2} K_{1/2+iE/2}(arg)]`` — one Bessel evaluation,
    using ``K_{1/2-iE/2} = conj(K_{1/2+iE/2})`` for real ``E``. Returns an ``mpf``
    at the active precision so the sign survives the ``e^{-pi E/4}`` envelope at
    height (eq. 15).
    """
    nu = mp.mpc(mp.mpf(1) / 2, mp.mpf(E) / 2)
    kp = mp.besselk(nu, arg)
    half = mp.mpf(theta) / 2
    return 2 * (mp.e ** (mp.mpc(0, -half)) * kp).real


def _energy_for_count(target: float) -> float:
    """Height ``E`` above the classical floor with ``N_bar(E) = target``.

    ``N_bar`` is monotone increasing for ``E > 2*pi`` (and the floor is ``4*pi``),
    so a plain float64 bisection is enough — this only ever sizes scans or builds
    the smooth average-zero reference, neither of which needs extended precision.
    """
    lo, hi = CLASSICAL_BOUND, 2.0 * CLASSICAL_BOUND
    while float(smooth_count(hi)) < target:
        hi *= 1.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(smooth_count(mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _energy_for_index(n: int) -> float:
    """Approximate height ``E`` of the ``n``-th eigenvalue: invert ``N_bar(E)=n``."""
    return _energy_for_count(float(n))


def required_dps(n: int, *, guard: int = 15) -> int:
    """Working precision so ``Xi`` stays sign-resolvable up to the ``n``-th level.

    ``|Xi(E)| ~ e^{-pi E / 4}`` (eq. 15), so at the top height ``E_n`` the value
    sits ``pi E_n / (4 ln 10)`` digits below O(1); we keep ``guard`` digits past
    that so the sign-scan is reliable.
    """
    e_max = _energy_for_index(n)
    lost = np.pi * e_max / 4.0 / LN10
    return int(np.ceil(lost)) + guard


def secular_spectrum(
    n: int,
    *,
    theta: float = THETA_RIEMANN,
    dps: int | None = None,
    e_lo: float = 1.0,
    scan_step: float = 0.5,
    e_pad: float = 1.2,
    cache_dir: Path | None = DATA,
    refresh: bool = False,
) -> np.ndarray:
    """First ``n`` positive eigenvalues ``E_1 < ... < E_n`` (the secular roots).

    Scans ``Xi`` with step ``scan_step`` at precision ``dps`` (chosen to keep the
    sign right under the envelope), brackets each sign change, and refines it. The
    spectrum is symmetric in ``+-E`` (eq. 14 with ``theta != pi`` has no zero
    mode), so the positive roots are the average-zero analogues. Results are cached
    to CSV keyed by ``(theta, n, dps)`` so reruns are instant.
    """
    if dps is None:
        dps = required_dps(n)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / (
            f"deformed_xp_theta{float(theta):g}_n{n}_dps{dps}.csv"
        )
        if cache.exists() and not refresh:
            vals = np.atleast_1d(np.loadtxt(cache))
            if vals.size >= n:
                return vals[:n]

    e_max = _energy_for_index(n) * e_pad
    roots: list[float] = []
    with mp.workdps(dps):
        step = mp.mpf(scan_step)
        e0 = mp.mpf(e_lo)
        f0 = secular(e0, theta)
        while e0 < e_max and len(roots) < n:
            e1 = e0 + step
            f1 = secular(e1, theta)
            if f0 == 0:
                roots.append(float(e0))
            elif f0 * f1 < 0:
                root = mp.findroot(
                    lambda E: secular(E, theta), (e0, e1), solver="anderson"
                )
                roots.append(float(root))
            e0, f0 = e1, f1

    out = np.asarray(roots[:n], dtype=np.float64)
    if cache is not None and out.size:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(cache, out)
    return out


def average_zeros(n: int) -> np.ndarray:
    """The first ``n`` *average* zero positions: ``E`` with ``N_bar(E) = k - 1/2``.

    The smooth (Pólya / Riemann–von Mangoldt) heights the deformed-``xp`` spectrum
    is meant to track. Built from the smooth count only — no actual zeros consumed
    — so it is a fair forward target for ``secular_spectrum``.
    """
    return np.asarray(
        [_energy_for_count(k - 0.5) for k in range(1, n + 1)], dtype=np.float64
    )


def analytic_unfold(spectrum: np.ndarray) -> np.ndarray:
    """Unfold the spectrum by the analytic smooth count ``N_bar(E)`` (eq. 2).

    ``N(E)`` is analytic for this model (unlike the De Bruijn–Newman empirical
    case), so the levels unfold by the exact Riemann–von Mangoldt smooth term and
    ``np.diff`` has mean ~1 — the unit-mean input the companion statistics (#24)
    consume. The mean-density agreement being *exact* is what makes the absence of
    GUE fluctuations the actual result.
    """
    return smooth_count(np.asarray(spectrum, dtype=np.float64))
