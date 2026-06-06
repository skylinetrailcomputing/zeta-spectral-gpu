"""Berry-Keating x<->p-symmetric deformed-``xp`` model: semiclassical count (#59).

German Sierra, *The Riemann zeros as spectrum and the Riemann hypothesis*
(Symmetry 11(4), 494, 2019; arXiv:1601.01797), section V.

The ``x<->p``-symmetric sibling of Sierra & Rodriguez-Laguna's
``H_I = x(p + l_p^2/p)`` (:mod:`zeta_spectral_gpu.deformed_xp`) is Berry &
Keating's

    H_II = (x + l_x^2/x)(p + l_p^2/p),    x, p > 0                       (eq. 5.4)

which restores the ``x<->p`` exchange symmetry that ``H_I`` breaks. Both are
members of the general family ``H = U(x) p + l_p^2 V(x)/p`` (eq. 5.5); ``H_II`` is
the one with ``U = V = x + l_x^2/x`` on the whole half-line ``(0, inf)``.

**Why there is no secular reference here** (the #23 -> #31 template does *not*
carry over). ``H_I``'s closed-form Bessel-``K`` secular equation (eq. 5.14) exists
because its associated 1+1D metric is *flat* (Rindler; scalar curvature ``R = 0``).
``H_II``'s metric is *curved*,

    R(x) = -4 l_x^2 / (x (x^2 + l_x^2)),                                (eq. 6.7)

so the eigenproblem is not exactly solvable: no cached source gives an ``H_II``
secular equation, and the change of variables that linearises ``H_I``'s local term
(``tau = log sqrt(x^2 + l_x^2)``) leaves ``H_II``'s nonlocal kernel with a
non-exponential factor, so it does *not* reduce to the Bessel problem. Sierra
himself drops back to the flat ``H_I`` for the solvable Dirac-ization "because the
flatness ... makes the computations easier" (sec. VII).

**What is exactly computable** -- and is the forward content of this module -- is
the semiclassical (Bohr-Sommerfeld) counting function: the number of levels below
``E`` is the phase-space area enclosed by the classical orbit ``H_II = E`` over
``2 pi hbar`` (``hbar = 1`` here). The fixed point is ``(l_x, l_p)``, the orbits
are closed loops around it, and the classical energy floor is
``H_II(l_x, l_p) = 4 l_x l_p = 4h`` (contrast ``H_I``'s ``2h``). For each ``x`` the
orbit spans ``p in (p_-, p_+)`` with ``p_+ - p_- = sqrt((E/f(x))^2 - 4 l_p^2)``,
``f(x) = x + l_x^2/x``, so the enclosed area is

    A(E) = integral sqrt((E/f(x))^2 - 4 l_p^2) dx   over { f(x) <= E/(2 l_p) }.

The substitution ``x = l_x e^theta`` (so ``f = 2 l_x cosh theta`` and the odd part
of ``dx`` drops out) collapses this to a one-dimensional integral that depends on
``l_x, l_p`` only through ``h = l_x l_p`` (the model's scaling symmetry, exactly as
for ``H_I``):

    A(E) = 4h integral_0^{arccosh B} sqrt(B^2 - cosh^2 theta) d theta,   B = E/(4h).

Its large-``E`` asymptotics (derived from the area; validated numerically in the
tests against Sierra eq. 5.18) are

    n_II(E) = A(E)/(2 pi) ~ (E / 2 pi)(log(E / h) - 1) + ...             (eq. 5.18)

-- the *same* two leading terms, at the *same* scale ``h = l_x l_p``, as ``H_I``
(eq. 5.17, ``n_I(E) ~ (E/2 pi)(log(E/h) - 1) - 1/2``) and as the average (smooth
Riemann-von Mangoldt) zero count ``N_bar(E) = (E/2 pi)(log(E/2 pi) - 1) + 7/8`` at
``h = 2 pi`` (:func:`zeta_spectral_gpu.zeros.smooth_count`). Two consequences, both
forward and both clean:

  * Restoring the ``x<->p`` symmetry does **not** change the mean spectral density:
    ``H_II`` reproduces the average zeros' two leading terms with *no* rescaling, at
    the same ``l_x l_p = 2 pi`` as ``H_I``. (The semiclassical area pins those two
    terms but not the ``O(1)`` constant -- the Maslov index and the ``7/8`` alike --
    so ``n_II - N_bar -> -7/8`` under the no-Maslov convention used here.)
  * What the deformation *does* change is everything below the leading density: the
    classical floor rises from ``2h`` (``H_I``) to ``4h``, and the subleading
    corrections differ (eq. 5.17 vs 5.18). The verdict is unchanged -- average
    density yes, Riemann fluctuations no (Sierra: "no trace of the exact Riemann
    zeros in the spectrum of the modified xp models"). This is the semiclassical
    companion to the ``H_I`` picket-fence result (#24); ``H_II``'s full quantum
    spectrum would need a direct diagonalisation of the curved-metric operator
    (out of scope here -- see the discussion above).

Forward, not inverse: a geometric deformation of ``xp``; no primes and no zeros are
consumed. The zeros appear only downstream, as the smooth-count comparison target.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from .deformed_xp import BESSEL_ARG
from .zeros import smooth_count

TWO_PI = 2.0 * np.pi

# Scaling symmetry: only the product h = l_x l_p enters the count. Use H_I's value
# (2 pi): H_II reproduces the average zeros' two leading terms at this same scale,
# with no rescaling (Sierra eq. 5.18).
H_PRODUCT = float(BESSEL_ARG)


def classical_bound(h: float = H_PRODUCT) -> float:
    """Classical energy floor ``H_II(l_x, l_p) = 4h``; no levels below it (eq. 5.4)."""
    return 4.0 * h


def classical_area(E: float, *, h: float = H_PRODUCT, dps: int = 30) -> float:
    """Phase-space area enclosed by the orbit ``H_II = E`` (``hbar = 1``).

    ``A(E) = 4h integral_0^{arccosh B} sqrt(B^2 - cosh^2 theta) d theta``,
    ``B = E / 4h``; zero at and below the floor ``E = 4h``. The integrand has an
    integrable square-root singularity at the upper endpoint, which mpmath's
    tanh-sinh :func:`mpmath.quad` resolves.
    """
    if E <= 4.0 * h:
        return 0.0
    with mp.workdps(dps):
        b = mp.mpf(E) / (4 * h)
        theta_max = mp.acosh(b)

        def integrand(t):
            # Clamp the integrable upper endpoint, where round-off can drive the
            # radicand slightly negative, so the quadrature result stays real.
            radicand = b * b - mp.cosh(t) ** 2
            return mp.sqrt(radicand) if radicand > 0 else mp.mpf(0)

        area = 4 * h * mp.quad(integrand, [0, theta_max])
    return float(area)


def classical_count(E: float, *, h: float = H_PRODUCT, dps: int = 30) -> float:
    """Semiclassical (Bohr-Sommerfeld) number of levels in ``(0, E)``: ``A(E)/2 pi``."""
    return classical_area(E, h=h, dps=dps) / TWO_PI


def leading_count(E: float, *, h: float = H_PRODUCT) -> float:
    """Leading large-``E`` asymptotic of the ``H_II`` count: ``(E/2 pi)(log(E/h) - 1)``.

    The closed-form leading term of :func:`classical_count` (Sierra eq. 5.18 with
    ``l_x l_p = h``); the two agree to a vanishing relative error at height, which is
    what pins the scale (``E/h``, the *same* as ``H_I``) independently of the
    quadrature -- see the tests.
    """
    return (E / TWO_PI) * (np.log(E / h) - 1.0)


def leading_count_asymmetric(E: float, *, h: float = H_PRODUCT) -> float:
    """``H_I``'s count for contrast (Sierra eq. 5.17): ``(E/2 pi)(log(E/h) - 1) - 1/2``."""
    return (E / TWO_PI) * (np.log(E / h) - 1.0) - 0.5


def average_count(E: float | np.ndarray) -> np.ndarray:
    """Average (smooth Riemann-von Mangoldt) zero count ``N_bar(E)``.

    The forward comparison target -- built from the smooth term only
    (:func:`zeta_spectral_gpu.zeros.smooth_count`), so no actual zeros are consumed.
    """
    return smooth_count(np.asarray(E, dtype=np.float64))


__all__ = [
    "H_PRODUCT",
    "average_count",
    "classical_area",
    "classical_bound",
    "classical_count",
    "leading_count",
    "leading_count_asymmetric",
]
