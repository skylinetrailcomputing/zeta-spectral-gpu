"""Davenport-Heilbronn negative control for the forward machinery (#85).

Every other experiment in the repo runs the forward statistics against objects
believed to satisfy RH and GUE universality, so a pipeline bug that *always
reports* "on-line / GUE" would never be caught. The Davenport-Heilbronn function
(1936) is the canonical falsifier: the period-5 Dirichlet series

    f(s) = sum_{n>=1} b(n) n^{-s},   b = (1, kappa, -kappa, -1, 0)  (period 5),

equivalently ``f = ((1 - i kappa) L(s, chi) + (1 + i kappa) L(s, chibar)) / 2``
with ``chi`` the odd complex character mod 5 and ``kappa`` the classical
algebraic constant fixed by the functional equation. f satisfies the exact
Riemann-type functional equation ``Lambda(s) = Lambda(1 - s)`` with
``Lambda(s) = (5/pi)^{(s+1)/2} Gamma((s+1)/2) f(s)`` — but it has **no Euler
product**, and it provably has zeros off the critical line (Davenport-Heilbronn
1936; Spira 1994 and Balanzario & Sanchez-Ortiz 2007 compute them). Running the
repo's forward machinery on f is therefore a control: the inputs are two
Dirichlet characters and one algebraic constant — pure number theory, the same
ingredients as :mod:`dirichlet` — and f's zeros (on- and off-line) are computed
as *output*, never consumed. See ``knowledge/davenport-heilbronn-control.md``.

Three control readouts live on top of this module:

* **growth** — the #43 RH-by-contradiction growth dichotomy on a *genuine*
  off-line zero (not a planted one): partial sums of the Dirichlet inverse of f
  at ``z = 1/2 + iE`` grow like ``n^{sigma_c - 1/2}`` when ``sigma_c + iE`` is
  an off-line zero (:func:`dirichlet_inverse`, :func:`growth_exponent`).
* **stats** — spacing / ratio statistics of f's critical-line zeros against the
  Bombieri-Hejhal expectation: a linear combination of two L-functions inherits
  the **superposition** of their (independent GUE) zero processes, not a single
  GUE (:func:`critical_line_zeros` for f, ``chi`` and ``chibar`` separately).
* **locator** — the #42 mirror locator driven by the Dirichlet-inverse weights
  of ``1/f``: with no Euler product the weights are not ``chi * mu`` and the
  clean peak-per-zero structure degrades (scored in the runner script).

Precision: fp64 throughout the scans (Euler-Maclaurin Hurwitz zeta, the #55
"modest height" regime — no Riemann-Siegel needed below ``t ~ 10^4``), with
mpmath as the independent cross-check and for polishing off-line roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import loggamma

from .dirichlet import dirichlet_character

MODULUS = 5

# Folded GUE / Poisson mean spacing-ratio references (Atas et al.), repeated here
# for the readout tables; the superposition reference is *computed*, not quoted.
MEAN_RATIO_GUE = 0.6027
MEAN_RATIO_POISSON = 2.0 * np.log(2.0) - 1.0


def gauss_sum(char: np.ndarray) -> complex:
    """``tau(chi) = sum_a chi(a) e^{2 pi i a / q}`` for a period-``q`` character."""
    char = np.asarray(char, dtype=np.complex128)
    q = char.size
    a = np.arange(q)
    return complex(np.sum(char * np.exp(2j * np.pi * a / q)))


@lru_cache(maxsize=1)
def dh_kappa() -> float:
    """The Davenport-Heilbronn constant ``kappa``, derived from the Gauss sum.

    The odd character mod 5 has root number ``eps = tau(chi) / (i sqrt 5)`` of
    modulus 1; writing ``eps = e^{2 i alpha}``, the combination
    ``(1 - i kappa) L_chi + (1 + i kappa) L_chibar`` is self-dual
    (``Lambda(s) = Lambda(1-s)``) exactly when ``(1 - i kappa) eps = 1 + i kappa``,
    i.e. ``kappa = tan(alpha)``. A test pins this against the classical closed
    form ``kappa = (sqrt(10 - 2 sqrt 5) - 2) / (sqrt 5 - 1)`` and the functional
    equation itself.
    """
    chi = dirichlet_character(MODULUS, 1)
    eps = gauss_sum(chi) / (1j * np.sqrt(MODULUS))
    alpha = np.angle(eps) / 2.0
    kappa = float(np.tan(alpha))
    if kappa < 0:  # chi vs chibar is a labelling choice; fix the classical sign
        kappa = -kappa
    return kappa


def dh_coefficient_pattern() -> np.ndarray:
    """Period-5 coefficients ``b(0..4) = (0, 1, kappa, -kappa, -1)`` (float64)."""
    k = dh_kappa()
    return np.array([0.0, 1.0, k, -k, -1.0])


def dh_coefficients(n: int) -> np.ndarray:
    """``b(1..n)`` as float64 (index 0 unused/0) — the only arithmetic input."""
    pattern = dh_coefficient_pattern()
    out = pattern[np.arange(n + 1) % MODULUS]
    out[0] = 0.0
    return out


def dh_value(s, *, dps: int = 25):
    """``f(s)`` via mpmath (analytic continuation included) — the reference path."""
    import mpmath as mp

    mp.mp.dps = dps
    coeffs = [mp.mpf(float(b)) for b in dh_coefficient_pattern()]
    return mp.dirichlet(s, coeffs)


def dh_completed(s, *, dps: int = 30):
    """``Lambda(s) = (5/pi)^{(s+1)/2} Gamma((s+1)/2) f(s)`` via mpmath.

    The odd-character completion; ``Lambda(s) = Lambda(1-s)`` is the functional
    equation the :func:`dh_kappa` derivation is tested against.
    """
    import mpmath as mp

    mp.mp.dps = dps
    s = mp.mpc(s)
    return (
        mp.power(mp.mpf(MODULUS) / mp.pi, (s + 1) / 2)
        * mp.gamma((s + 1) / 2)
        * dh_value(s, dps=dps)
    )


# --- fp64 Euler-Maclaurin Hurwitz zeta (the modest-height evaluator) ----------


@lru_cache(maxsize=1)
def _bernoulli_over_factorial(tail: int = 12) -> tuple[float, ...]:
    """``B_{2j} / (2j)!`` for ``j = 1..tail`` (exact via mpmath, cached)."""
    import mpmath as mp

    return tuple(
        float(mp.bernoulli(2 * j) / mp.factorial(2 * j)) for j in range(1, tail + 1)
    )


def hurwitz_zeta_em(
    s: np.ndarray, a: float, *, n_terms: int | None = None, tail: int = 12
) -> np.ndarray:
    """``zeta(s, a)`` for an array of complex ``s`` by Euler-Maclaurin (fp64).

    Plain numpy: the truncated sum to ``N = n_terms``, the ``(N+a)^{1-s}/(s-1)``
    integral term, and ``tail`` Bernoulli corrections. The corrections decay like
    ``(|s| / 2 pi (N+a))^{2j}``, so ``N`` defaults to ``0.7 max|Im s| + 32`` —
    comfortably convergent for the modest heights this control runs at (a test
    pins fp64 agreement with ``mpmath.zeta(s, a)``). Valid on the whole plane
    except the pole ``s = 1``.
    """
    s = np.asarray(s, dtype=np.complex128)
    if n_terms is None:
        t_max = float(np.max(np.abs(s.imag))) if s.size else 0.0
        n_terms = int(0.7 * t_max) + 32
    flat = s.reshape(-1)

    total = np.zeros(flat.shape, dtype=np.complex128)
    chunk = max(1, int(2**22 / max(flat.size, 1)) + 1)
    for k0 in range(0, n_terms, chunk):
        k = np.arange(k0, min(k0 + chunk, n_terms), dtype=np.float64) + a
        total += np.exp(-np.multiply.outer(flat, np.log(k))).sum(axis=1)

    edge = n_terms + a  # N + a
    log_edge = np.log(edge)
    edge_pow = np.exp(-flat * log_edge)  # (N+a)^{-s}
    total += edge_pow * edge / (flat - 1.0)  # (N+a)^{1-s} / (s-1)
    total += 0.5 * edge_pow

    poch = flat.copy()  # s (s+1) ... (s + 2j - 2), iterated
    scale = edge_pow / edge  # (N+a)^{-s - (2j-1)}, iterated
    for j, b2j in enumerate(_bernoulli_over_factorial(tail), start=1):
        total += b2j * poch * scale
        poch = poch * (flat + (2 * j - 1)) * (flat + 2 * j)
        scale = scale / (edge * edge)
    return total.reshape(s.shape)


def dh_theta(t: np.ndarray | float) -> np.ndarray | float:
    """Phase of the completed gamma factor on the line (the unfolding clock).

    ``theta_5(t) = Im log Gamma(3/4 + it/2) + (t/2) log(5/pi)`` — the odd-mod-5
    analogue of the Riemann-Siegel theta. ``e^{i theta_5(t)} f(1/2 + it)`` is
    real (the Hardy-Z analogue), and ``theta_5(t)/pi`` is the smooth zero count
    used to unfold.
    """
    t = np.asarray(t, dtype=np.float64)
    val = loggamma(0.75 + 0.5j * t).imag + 0.5 * t * np.log(MODULUS / np.pi)
    return val if val.shape else float(val)


def dh_smooth_count(t: np.ndarray | float) -> np.ndarray | float:
    """Smooth counting function ``N_f(t) ~ theta_5(t)/pi + 1`` (unfolding map)."""
    return dh_theta(t) / np.pi + 1.0


@dataclass(frozen=True)
class LineScan:
    """One fp64 pass over a critical-line grid: f and the two component L's.

    ``z_*`` are the real rotated Hardy-Z analogues (sign changes = on-line
    zeros); ``residual`` is the largest imaginary part left after rotation, a
    cheap end-to-end fp64 health check (should sit at rounding level).
    """

    grid: np.ndarray
    z_f: np.ndarray
    z_chi: np.ndarray
    z_chibar: np.ndarray
    residual: float


def line_scan(t: np.ndarray, *, n_terms: int | None = None) -> LineScan:
    """Evaluate ``Z_f``, ``Z_chi``, ``Z_chibar`` on a grid of ordinates (fp64).

    One set of four Hurwitz-zeta arrays ``zeta(s, a/5)`` feeds all three
    functions: f through its real period-5 coefficients, the component
    L-functions through ``chi`` / ``chibar`` with the half-root-number rotation
    ``eps^{-1/2}`` that makes each ``Z`` real on the line. All three are scanned
    by the *same* code path — that is the point of the control.
    """
    t = np.asarray(t, dtype=np.float64)
    s = 0.5 + 1j * t
    zetas = [hurwitz_zeta_em(s, a / MODULUS, n_terms=n_terms) for a in range(1, 5)]
    prefactor = np.exp(-s * np.log(MODULUS))  # 5^{-s}

    chi = dirichlet_character(MODULUS, 1)
    b = dh_coefficient_pattern()
    f_vals = prefactor * sum(b[a] * zetas[a - 1] for a in range(1, 5))
    l_chi = prefactor * sum(chi[a] * zetas[a - 1] for a in range(1, 5))
    l_chibar = prefactor * sum(np.conj(chi[a]) * zetas[a - 1] for a in range(1, 5))

    rotation = np.exp(1j * dh_theta(t))
    eps = gauss_sum(chi) / (1j * np.sqrt(MODULUS))
    half_root = np.exp(-0.5j * np.angle(eps))

    rotated = (
        rotation * f_vals,
        rotation * half_root * l_chi,
        rotation * np.conj(half_root) * l_chibar,
    )
    residual = max(float(np.max(np.abs(r.imag))) for r in rotated)
    return LineScan(
        grid=t,
        z_f=rotated[0].real,
        z_chi=rotated[1].real,
        z_chibar=rotated[2].real,
        residual=residual,
    )


def _bisect_sign_changes(
    z_of_t, lo: np.ndarray, hi: np.ndarray, *, iters: int = 47
) -> np.ndarray:
    """Vectorised bisection of bracketed sign changes of a real callable."""
    lo = lo.copy()
    hi = hi.copy()
    z_lo = z_of_t(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        z_mid = z_of_t(mid)
        left = np.signbit(z_lo) != np.signbit(z_mid)
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        z_lo = np.where(left, z_lo, z_mid)
    return 0.5 * (lo + hi)


def critical_line_zeros(
    t_lo: float,
    t_hi: float,
    *,
    step: float = 0.05,
    which: str = "f",
    n_terms: int | None = None,
) -> np.ndarray:
    """On-line zero ordinates of ``f`` (or ``chi`` / ``chibar``) in ``(t_lo, t_hi)``.

    Sign changes of the rotated real ``Z`` on a fixed grid, refined by
    vectorised bisection — the #55 sign-scan pattern at modest height. Forward:
    the ordinates are *produced* here (characters and ``kappa`` in, zeros out)
    and only ever compared downstream. Off-line zeros do not change ``Z_f``'s
    sign and are deliberately not found by this scan; :func:`off_line_zeros`
    handles them, and the deficit against :func:`dh_smooth_count` is part of
    the readout.
    """
    grid = np.arange(t_lo, t_hi + step, step)
    if n_terms is None:
        n_terms = int(0.7 * float(t_hi)) + 32

    def z_of_t(t: np.ndarray) -> np.ndarray:
        scan = line_scan(t, n_terms=n_terms)
        return {"f": scan.z_f, "chi": scan.z_chi, "chibar": scan.z_chibar}[which]

    z = z_of_t(grid)
    flip = np.signbit(z[:-1]) != np.signbit(z[1:])
    return _bisect_sign_changes(z_of_t, grid[:-1][flip], grid[1:][flip])


def off_line_zeros(
    t_lo: float,
    t_hi: float,
    *,
    sigma_lo: float = 0.55,
    sigma_hi: float = 1.05,
    grid_step: float = 0.05,
    sigma_step: float = 0.025,
    dps: int = 30,
    f_tol: float = 1e-20,
) -> np.ndarray:
    """Zeros of f with ``Re s > 1/2`` in the box — the genuine RH violations.

    fp64 grid of ``|f|`` over the box marks candidate local minima; each is
    polished with mpmath ``findroot`` (complex secant on :func:`dh_value`) and
    kept only if it is a verified zero (``|f| < f_tol`` at ``dps`` digits)
    inside the box. By the functional equation and real coefficients each zero
    found here has mirrors at ``1 - rho`` and the conjugates; only the
    ``sigma > 1/2, t > 0`` representative is returned. Output, never input.
    """
    import mpmath as mp

    sigmas = np.arange(sigma_lo, sigma_hi + sigma_step, sigma_step)
    ts = np.arange(t_lo, t_hi + grid_step, grid_step)
    s = sigmas[:, None] + 1j * ts[None, :]
    n_terms = int(0.7 * float(t_hi)) + 32
    zetas = [
        hurwitz_zeta_em(s, a / MODULUS, n_terms=n_terms) for a in range(1, MODULUS)
    ]
    b = dh_coefficient_pattern()
    f_abs = np.abs(
        np.exp(-s * np.log(MODULUS)) * sum(b[a] * zetas[a - 1] for a in range(1, 5))
    )

    interior = np.ones_like(f_abs, dtype=bool)
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    minima = interior.copy()
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        minima &= f_abs <= np.roll(f_abs, shift, axis=axis)
    candidates = np.argwhere(minima & (f_abs < 0.2))

    mp.mp.dps = dps
    found: list[complex] = []
    for i, j in candidates:
        seed = complex(s[i, j])
        try:
            root = mp.findroot(lambda z: dh_value(z, dps=dps), mp.mpc(seed))
        except (ValueError, ZeroDivisionError):
            continue
        # Verify zero-ness at full precision BEFORE rounding: at the fp64-rounded
        # point the residual is |f'| * O(1e-14), which would mask a genuine zero.
        if abs(dh_value(root, dps=dps)) >= f_tol:
            continue
        root = complex(root)
        if not (sigma_lo <= root.real <= sigma_hi and t_lo <= root.imag <= t_hi):
            continue
        if any(abs(root - r) < 1e-6 for r in found):
            continue
        found.append(root)
    return np.array(sorted(found, key=lambda z: (z.imag, z.real)))


# --- Dirichlet inverse + growth dichotomy (readouts 1 and 3) ------------------


def dirichlet_inverse(n: int) -> np.ndarray:
    """Coefficients ``c(1..n)`` of ``1/f`` by Dirichlet-series inversion (float64).

    ``c(1) = 1`` and ``c(m) = -sum_{d | m, d > 1} b(d) c(m/d)`` — the analogue of
    the Moebius weights ``chi mu`` that the locator uses for a genuine
    L-function, except f has no Euler product so ``c`` is *not* multiplicative
    and (because f has zeros with ``Re s > 1``) not even bounded. The ascending
    sieve distributes each finalised ``c(q)`` to its multiples; O(n log n).
    """
    b = dh_coefficient_pattern()
    c = np.zeros(n + 1)
    if n >= 1:
        c[1] = 1.0
    for q in range(1, n // 2 + 1):
        if c[q] == 0.0:
            continue
        d = np.arange(2, n // q + 1)
        c[q * d] -= b[d % MODULUS] * c[q]
    return c


def partial_sum_profile(
    E: float,
    weights: np.ndarray,
    truncations: np.ndarray,
    *,
    sigma: float = 0.5,
) -> np.ndarray:
    """``|sum_{k<=n} w(k) k^{-(sigma + iE)}|`` at each truncation ``n``.

    The #43 growth object with general weights: pass ``dirichlet_inverse`` for f
    (or ``chi * mu`` for a genuine L-function). One cumulative sum, sampled.
    """
    truncations = np.asarray(truncations, dtype=np.int64)
    n_top = int(truncations.max())
    k = np.arange(1, n_top + 1, dtype=np.float64)
    terms = weights[1 : n_top + 1] * k**-sigma * np.exp(-1j * E * np.log(k))
    return np.abs(np.cumsum(terms))[truncations - 1]


def loglog_rms_slope(truncations: np.ndarray, values: np.ndarray) -> float:
    """Log-log slope of the RMS of ``values`` in geometric truncation bins.

    The growth-law discriminator (#43): at an off-line zero ``sigma_c + iE`` the
    profile grows like ``n^{sigma_c - 1/2}`` (slope ``sigma_c - 1/2``); at an
    on-line zero it grows only logarithmically (slope ~ 0 on this measure); off
    any zero it stays bounded. RMS-in-bins smooths the oscillation before the
    fit, exactly as the #43 demo does.
    """
    truncations = np.asarray(truncations, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    edges = np.logspace(
        np.log10(truncations.min()), np.log10(truncations.max() + 1), 11
    )
    centres, rms = [], []
    for a, lim in zip(edges[:-1], edges[1:]):
        mask = (truncations >= a) & (truncations < lim)
        if not np.any(mask):
            continue
        centres.append(np.sqrt(a * lim))
        rms.append(np.sqrt(np.mean(values[mask] ** 2)))
    return float(np.polyfit(np.log(centres), np.log(rms), 1)[0])


def growth_exponent(
    E: float,
    weights: np.ndarray,
    *,
    n_max: int,
    n_min: int = 100,
    sigma: float = 0.5,
) -> float:
    """Measured growth slope of the partial-sum profile at energy ``E``."""
    truncations = np.unique(
        np.logspace(np.log10(n_min), np.log10(n_max), 400).astype(np.int64)
    )
    profile = partial_sum_profile(E, weights, truncations, sigma=sigma)
    return loglog_rms_slope(truncations, profile)
