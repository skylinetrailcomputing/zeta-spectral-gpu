"""Arithmetic (beyond-GUE) correlations of the Riemann zeros — CPU reference.

Issue #84. Every statistic in the repo so far checks the zeros against a
*universal* random-matrix prediction; this module computes the first place the
zeros are *zeta* rather than generic GUE: the lower-order terms of the pair
correlation and the prime peaks of the spectral form factor. Two sides:

* **Prediction** — assembled purely from the primes and ``zeta`` on the 1-line:
  the Conrey–Snaith ratios-conjecture pair correlation (Theorem 4.1 of
  *Applications of the L-functions ratios conjectures*, Proc. LMS 94, 2007),
  its Bogomolny–Keating Hardy–Littlewood form (an exactly equal regrouping,
  asserted in the tests), and the explicit-formula (Landau) prime-peak
  prediction for the zero Fourier transform. No zeros are consumed.
* **Empirical** — the zeros enter only as the *output being characterised*:
  raw-separation pair histograms and the windowed Fourier statistic
  ``S(u) = sum_n w(tau_n) e^{i u tau_n}``.

See ``knowledge/arithmetic-correlations.md`` for the equation-by-equation
derivation and source pinning. GPU mirror: ``arithmetic_correlations_gpu.py``.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# Default prime cutoff for the convergent Euler products / prime sums A, B,
# Phi. Their per-prime factors are 1 + O(1/p^2), so the truncation error is
# ~ sum_{p > P} 1/p^2 ~ 1/(P log P) — far below plotting resolution at 1e5.
DEFAULT_P_MAX = 100_000


# --- Prime-side inputs -------------------------------------------------------


def primes_upto(n: int) -> np.ndarray:
    """Primes ``<= n`` (int64) by a plain odd sieve of Eratosthenes."""
    if n < 2:
        return np.empty(0, dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def von_mangoldt(n_max: int) -> np.ndarray:
    """``Lambda(n)`` for ``n = 0..n_max`` (index = n; entries 0 and 1 are 0).

    ``Lambda(p^m) = log p`` at prime powers, 0 elsewhere — the arithmetic
    weight of the explicit formula. Pure prime-side input (forward).
    """
    lam = np.zeros(n_max + 1, dtype=np.float64)
    for p in primes_upto(n_max):
        logp = np.log(float(p))
        q = int(p)
        while q <= n_max:
            lam[q] = logp
            q *= int(p)
    return lam


# --- Conrey–Snaith arithmetic factors A and B --------------------------------
#
# Conrey & Snaith 2007, eqs. (4.20)-(4.21), evaluated on the imaginary axis
# eta = i*eps (eps = raw ordinate separation):
#
#   A(eta) = prod_p (1 - p^{-1-eta}) (1 - 2/p + p^{-1-eta}) / (1 - 1/p)^2
#   B(eta) = sum_p ( log p / (p^{1+eta} - 1) )^2


def a_factor(eps: np.ndarray | float, *, p_max: int = DEFAULT_P_MAX) -> np.ndarray:
    """Conrey–Snaith ``A(i*eps)`` on a real separation grid (complex128)."""
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    p = primes_upto(p_max).astype(np.float64)
    out = np.ones(eps.shape, dtype=np.complex128)
    # Chunk over primes to bound the (n_primes x n_eps) temporary.
    for lo in range(0, p.size, 2048):
        q = p[lo : lo + 2048, None]
        x = q ** (-1.0 - 1j * eps[None, :])  # p^{-1-i eps}
        factor = (1.0 - x) * (1.0 - 2.0 / q + x) / (1.0 - 1.0 / q) ** 2
        out *= factor.prod(axis=0)
    return out


def b_factor(eps: np.ndarray | float, *, p_max: int = DEFAULT_P_MAX) -> np.ndarray:
    """Conrey–Snaith ``B(i*eps)`` on a real separation grid (complex128)."""
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    p = primes_upto(p_max).astype(np.float64)
    out = np.zeros(eps.shape, dtype=np.complex128)
    for lo in range(0, p.size, 2048):
        q = p[lo : lo + 2048, None]
        term = (np.log(q) / (q ** (1.0 + 1j * eps[None, :]) - 1.0)) ** 2
        out += term.sum(axis=0)
    return out


def _zeta_one_line(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``zeta(1 + i*eps)`` and ``(zeta'/zeta)'(1 + i*eps)`` via mpmath.

    The 1-line is where the arithmetic prediction needs zeta; mpmath keeps the
    near-pole evaluations (small ``eps``) honest. Returns complex128 arrays.
    ``(zeta'/zeta)' = (zeta'' zeta - zeta'^2) / zeta^2`` from the derivatives.
    """
    import mpmath as mp

    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    z = np.empty(eps.shape, dtype=np.complex128)
    ddlog = np.empty(eps.shape, dtype=np.complex128)
    with mp.workdps(30):
        for i, e in enumerate(eps):
            s = mp.mpc(1.0, float(e))
            z0 = mp.zeta(s)
            z1 = mp.zeta(s, derivative=1)
            z2 = mp.zeta(s, derivative=2)
            z[i] = complex(z0)
            ddlog[i] = complex((z2 * z0 - z1 * z1) / (z0 * z0))
    return z, ddlog


# --- Pair-correlation predictions --------------------------------------------


def cs_integrand(
    eps: np.ndarray | float, t: float, *, p_max: int = DEFAULT_P_MAX
) -> np.ndarray:
    """Conrey–Snaith pair density ``rho_2(eps; t)`` at fixed height ``t``.

    The (off-diagonal) integrand of CS Theorem 4.1 divided by ``(2 pi)^2``:

        rho_2 = (1/4 pi^2) [ log^2(t/2pi) + 2 Re( (zeta'/zeta)'(1+i eps)
                + (t/2pi)^{-i eps} zeta(1-i eps) zeta(1+i eps) A(i eps)
                - B(i eps) ) ]

    Units: expected ordered pairs (gamma' < gamma) per unit height per unit
    separation. ``-> dbar(t)^2`` as ``eps -> infinity`` (decorrelation); the
    ``eps -> 0`` pole of each term cancels in the real part (principal value).
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    z, ddlog = _zeta_one_line(eps)
    a = a_factor(eps, p_max=p_max)
    b = b_factor(eps, p_max=p_max)
    logt = np.log(t / TWO_PI)
    osc = np.exp(-1j * eps * logt)  # (t/2pi)^{-i eps}
    g = ddlog + osc * np.conj(z) * z * a - b
    return (logt**2 + 2.0 * g.real) / (4.0 * np.pi**2)


def cs_pair_density(
    eps: np.ndarray | float,
    t_lo: float,
    t_hi: float,
    *,
    p_max: int = DEFAULT_P_MAX,
) -> np.ndarray:
    """CS prediction integrated exactly over the height window ``[t_lo, t_hi]``.

    Expected ordered pairs (``gamma' < gamma``, both in the window) per unit
    separation at separation ``eps``. The ``t``-dependence of Theorem 4.1 is
    elementary — ``log^2(t/2pi)`` and ``(t/2pi)^{-i eps}`` — so the window
    integral is closed-form; everything else is constant in ``t``:

        int log^2(t/2pi) dt = t (log^2(t/2pi) - 2 log(t/2pi) + 2)
        int (t/2pi)^{-i eps} dt = 2pi (t/2pi)^{1 - i eps} / (1 - i eps)
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    z, ddlog = _zeta_one_line(eps)
    a = a_factor(eps, p_max=p_max)
    b = b_factor(eps, p_max=p_max)

    def log2_primitive(t: float) -> float:
        lg = np.log(t / TWO_PI)
        return t * (lg * lg - 2.0 * lg + 2.0)

    def osc_primitive(t: float) -> np.ndarray:
        # 2pi (t/2pi)^{1-i eps} / (1 - i eps)
        return TWO_PI * np.exp((1.0 - 1j * eps) * np.log(t / TWO_PI)) / (1.0 - 1j * eps)

    dt = t_hi - t_lo
    log2_int = log2_primitive(t_hi) - log2_primitive(t_lo)
    osc_int = osc_primitive(t_hi) - osc_primitive(t_lo)
    g_int = ddlog * dt + osc_int * np.conj(z) * z * a - b * dt
    return (log2_int + 2.0 * g_int.real) / (4.0 * np.pi**2)


def bk_integrand(
    eps: np.ndarray | float,
    t: float,
    *,
    p_max: int = DEFAULT_P_MAX,
    m_max: int = 64,
) -> np.ndarray:
    """Bogomolny–Keating Hardy–Littlewood form of ``rho_2(eps; t)``.

    Independent transcription used to cross-check :func:`cs_integrand` — the
    two are *exactly* equal (the tests assert it):

        rho_2 = dbar^2 + R2_diag + R2_off
        R2_diag = -(1/4 pi^2) d^2/d eps^2 log( |zeta(1+i eps)|^2 Phi_diag )
        R2_off  = (1/4 pi^2) |zeta(1+i eps)|^2 e^{i eps log(t/2pi)} Phi_off + c.c.
        Phi_diag = exp( 2 sum_p sum_{m>=1} (1-m)/(m^2 p^m) cos(m eps log p) )
        Phi_off  = prod_p ( 1 - (1 - p^{i eps})^2 / (p-1)^2 )

    (Bogomolny, *Quantum and arithmetical chaos*, nlin/0312061, Lecture 2.)
    The zeta part of the diagonal derivative is analytic:
    ``d^2/d eps^2 log|zeta(1+i eps)|^2 = -2 Re (zeta'/zeta)'(1+i eps)``; the
    ``Phi_diag`` part differentiates the cosine series term by term.
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    z, ddlog = _zeta_one_line(eps)
    p = primes_upto(p_max).astype(np.float64)

    # diag_primes accumulates -d^2/d eps^2 of log Phi_diag (the m^2 in the
    # denominator cancels against the cosine's second derivative, and the
    # series' overall minus sign is folded into the (1-m) coefficient).
    diag_primes = np.zeros(eps.shape, dtype=np.float64)
    for m in range(1, m_max + 1):
        pm = p ** float(m)
        keep = pm < 1e18
        if not keep.any():
            break
        q, logq = pm[keep], np.log(p[keep])
        coeff = (1.0 - m) * logq**2 / q
        diag_primes += 2.0 * (
            coeff[:, None] * np.cos(m * eps[None, :] * logq[:, None])
        ).sum(axis=0)
    r2_diag = (2.0 * ddlog.real + diag_primes) / (4.0 * np.pi**2)

    # Phi_off Euler product, chunked like a_factor.
    phi_off = np.ones(eps.shape, dtype=np.complex128)
    for lo in range(0, p.size, 2048):
        q = p[lo : lo + 2048, None]
        x = q ** (1j * eps[None, :])  # p^{+i eps}
        phi_off *= (1.0 - (1.0 - x) ** 2 / (q - 1.0) ** 2).prod(axis=0)
    logt = np.log(t / TWO_PI)
    osc = np.exp(1j * eps * logt) * (np.conj(z) * z) * phi_off
    r2_off = 2.0 * osc.real / (4.0 * np.pi**2)

    dbar = logt / TWO_PI
    return dbar**2 + r2_diag + r2_off


def gue_pair_density(
    eps: np.ndarray | float, t_lo: float, t_hi: float, *, n_t: int = 257
) -> np.ndarray:
    """Universal (sine-kernel-only) counterpart of :func:`cs_pair_density`.

    ``rho_2^GUE(eps; t) = dbar(t)^2 (1 - sinc^2(dbar(t) eps))`` integrated over
    the same window (Simpson on a ``t`` grid — no closed form needed). The
    difference CS - GUE is the arithmetic content the experiment looks for.
    """
    from scipy.integrate import simpson

    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    t = np.linspace(t_lo, t_hi, n_t)
    dbar = np.log(t / TWO_PI) / TWO_PI
    vals = dbar[:, None] ** 2 * (1.0 - np.sinc(dbar[:, None] * eps[None, :]) ** 2)
    return simpson(vals, x=t, axis=0)


# --- Spectral form factor: the zero Fourier transform ------------------------
#
# S(u) = sum_n w(tau_n) e^{i u tau_n} over a height window. The explicit
# formula (Landau's theorem under a window) predicts prime-power peaks:
#
#   S(u) ~ (1/2pi) int w(t) log(t/2pi) e^{iut} dt
#          - (1/2pi) sum_{n>=2} Lambda(n)/sqrt(n) * W(log n - u)
#
# where W(y) = int w(t) e^{-iyt} dt is the window transform. The first term is
# the smooth density spike at u ~ 0; the second puts a peak of width ~1/(t_hi -
# t_lo) at every u = log p^m with von Mangoldt weight — primes literally
# visible in the Fourier statistics of the zeros.

_WINDOWS = ("rect", "hann")


def window_weights(tau: np.ndarray, t_lo: float, t_hi: float, kind: str) -> np.ndarray:
    """Taper weights ``w(tau)`` on ``[t_lo, t_hi]`` (0 outside)."""
    tau = np.asarray(tau, dtype=np.float64)
    inside = (tau >= t_lo) & (tau <= t_hi)
    if kind == "rect":
        return inside.astype(np.float64)
    if kind == "hann":
        phase = TWO_PI * (tau - t_lo) / (t_hi - t_lo)
        return np.where(inside, 0.5 * (1.0 - np.cos(phase)), 0.0)
    raise ValueError(f"unknown window {kind!r}; use one of {_WINDOWS}")


def window_transform(
    y: np.ndarray | float, t_lo: float, t_hi: float, kind: str
) -> np.ndarray:
    """Analytic ``W(y) = int_{t_lo}^{t_hi} w(t) e^{-iyt} dt`` (complex128).

    rect: ``W(y) = dT e^{-iy t_c} sinc(y dT / 2pi)`` (``t_c`` = window centre).
    hann: three shifted rect transforms,
    ``W_h(y) = W_r(y)/2 - e^{-2pi i t_lo/dT} W_r(y - 2pi/dT)/4
              - e^{+2pi i t_lo/dT} W_r(y + 2pi/dT)/4``.
    """
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    dt = t_hi - t_lo
    t_c = 0.5 * (t_lo + t_hi)

    def rect(yy: np.ndarray) -> np.ndarray:
        return dt * np.exp(-1j * yy * t_c) * np.sinc(yy * dt / TWO_PI)

    if kind == "rect":
        return rect(y)
    if kind == "hann":
        shift = TWO_PI / dt
        phase = np.exp(-1j * TWO_PI * t_lo / dt)
        return (
            0.5 * rect(y)
            - 0.25 * phase * rect(y - shift)
            - 0.25 * np.conj(phase) * rect(y + shift)
        )
    raise ValueError(f"unknown window {kind!r}; use one of {_WINDOWS}")


def window_l2(t_lo: float, t_hi: float, kind: str) -> float:
    """``int w(t)^2 dt`` — the form-factor normalisation of the window."""
    dt = t_hi - t_lo
    if kind == "rect":
        return dt
    if kind == "hann":
        return 0.375 * dt
    raise ValueError(f"unknown window {kind!r}; use one of {_WINDOWS}")


def zero_fourier(
    tau: np.ndarray,
    u: np.ndarray | float,
    t_lo: float,
    t_hi: float,
    *,
    window: str = "hann",
    chunk: int = 65_536,
) -> np.ndarray:
    """Empirical ``S(u) = sum_n w(tau_n) e^{i u tau_n}`` (complex128).

    CPU reference: chunked dense broadcast, O(N * len(u)). The GPU mirror in
    ``arithmetic_correlations_gpu.py`` must reproduce this to floating-point
    tolerance. The zeros are consumed only as the data being transformed.
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    tau = np.asarray(tau, dtype=np.float64)
    w = window_weights(tau, t_lo, t_hi, window)
    keep = w > 0.0
    tau, w = tau[keep], w[keep]
    out = np.zeros(u.shape, dtype=np.complex128)
    for lo in range(0, tau.size, chunk):
        t = tau[lo : lo + chunk]
        ww = w[lo : lo + chunk]
        out += (ww[None, :] * np.exp(1j * u[:, None] * t[None, :])).sum(axis=1)
    return out


def prime_prediction(
    u: np.ndarray | float,
    t_lo: float,
    t_hi: float,
    *,
    window: str = "hann",
    include_smooth: bool = True,
) -> np.ndarray:
    """Explicit-formula prediction for ``S(u)`` — primes in, no zeros.

    The prime-power sum runs to ``n <= exp(max u + spectral width)``; the
    smooth (density) term ``(1/2pi) int w(t) log(t/2pi) e^{iut} dt`` is kept to
    leading order in the slowly varying log, ``log(t_c/2pi) W(-u) / 2pi`` —
    a spike of width ~1/(t_hi - t_lo) at u = 0, negligible elsewhere.
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    margin = 64.0 * TWO_PI / (t_hi - t_lo)  # a few spectral widths
    n_max = int(np.exp(float(u.max()) + margin)) + 1
    lam = von_mangoldt(n_max)
    n = np.nonzero(lam)[0].astype(np.float64)
    weight = lam[np.nonzero(lam)] / np.sqrt(n)
    out = np.zeros(u.shape, dtype=np.complex128)
    for lo in range(0, n.size, 512):
        logn = np.log(n[lo : lo + 512])
        wt = weight[lo : lo + 512]
        shifts = window_transform(
            (logn[:, None] - u[None, :]).ravel(), t_lo, t_hi, window
        ).reshape(logn.size, u.size)
        out += (wt[:, None] * shifts).sum(axis=0)
    out *= -1.0 / TWO_PI

    if include_smooth:
        t_c = 0.5 * (t_lo + t_hi)
        out += np.log(t_c / TWO_PI) * window_transform(-u, t_lo, t_hi, window) / TWO_PI
    return out


def diagonal_ramp(
    u: np.ndarray | float, t_lo: float, t_hi: float, *, window: str = "hann"
) -> np.ndarray:
    """Smoothed GUE/diagonal background for ``|S(u)|^2``.

    Averaging the prime peaks over a few spectral widths gives the random-
    matrix ramp ``<|S|^2> = (u/2pi) int w^2`` (peaks at density du = 1/u with
    mean weight ``Lambda^2(n)/n``), saturating at the plateau ``sum_n w_n^2 ~
    dbar * int w^2`` at the Heisenberg frequency ``u = 2 pi dbar``. The
    arithmetic content of the form factor is exactly the spiky departure of
    ``|S|^2`` from this smooth curve.
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    l2 = window_l2(t_lo, t_hi, window)
    t_c = 0.5 * (t_lo + t_hi)
    dbar = np.log(t_c / TWO_PI) / TWO_PI
    return np.minimum(np.abs(u) / TWO_PI, dbar) * l2
