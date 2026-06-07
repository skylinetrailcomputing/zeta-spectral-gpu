"""Katz-Sarnak family statistics for quadratic Dirichlet L-functions (issue #51).

Where the warm-up statistics (:mod:`spacing`) probe the *single* sequence of zeta
zeros against the GUE (unitary) ensemble, Katz-Sarnak universality is a statement
about a whole **family** of L-functions: as the conductor grows, the low-lying
zeros of the family -- rescaled by the conductor -- follow the 1-level density of
one of the classical compact groups, and *which* group is fixed by the family's
symmetry type. The cleanest case is the family of quadratic Dirichlet
L-functions ``L(s, chi_d)`` over fundamental discriminants ``d``: its symmetry
type is **symplectic** (Katz-Sarnak; Ozluk-Snyder; Rubinstein 2001), so the
family-averaged 1-level density should track

    W_Sp(x) = 1 - sin(2 pi x) / (2 pi x)

which *vanishes at the central point* ``x = 0`` -- zeros are repelled from the
center -- distinguishing it from the unitary kernel ``W_U(x) = 1`` (flat) and the
even-orthogonal kernel ``W_SO+(x) = 1 + sin(2 pi x)/(2 pi x)`` (enhanced at the
center).

Forward, not inverse (see the README and :mod:`dirichlet`): each member's zeros
are produced *independently* (mpmath, via :func:`dirichlet.lfunction_zeros`) and
the symmetry kernel is the parameter-free **prediction** we compare against. The
only input is the discriminant ``d`` -- pure number theory in (the quadratic
character ``chi_d(n) = (d | n)``, a Kronecker symbol), the symmetry type out. No
zero is ever fit.

The character itself is the new number-theoretic input: the quadratic character
of a fundamental discriminant ``d`` is the Kronecker symbol ``n -> (d | n)``, a
real primitive character of conductor ``|d|``. Building it directly (rather than
through a primitive root, as :mod:`dirichlet` does for prime moduli) gives the
whole family -- every fundamental discriminant, real (``d > 0``) and imaginary
(``d < 0``) -- from one elementary routine, and it drops straight into the
period-array API the locator and ``mpmath`` already consume.
"""

from __future__ import annotations

import numpy as np

from .dirichlet import lfunction_zeros

PI = np.pi


# --- Quadratic characters: the family's number-theoretic input ----------------


def kronecker_symbol(a: int, n: int) -> int:
    """The Kronecker symbol ``(a | n)`` (an integer in ``{-1, 0, 1}``).

    Extends the Jacobi symbol to all integers ``n`` via the rules ``(a | 0) = 1``
    iff ``a = +-1``, ``(a | -1) = sign(a)``, and the supplement
    ``(a | 2) = 0`` if ``a`` even, ``+1`` if ``a == +-1 (mod 8)``, ``-1`` if
    ``a == +-3 (mod 8)``. Used to form the quadratic character of a fundamental
    discriminant, ``chi_d(n) = (d | n)``.
    """
    if n == 0:
        return 1 if a in (1, -1) else 0
    result = 1
    if n < 0:
        n = -n
        if a < 0:
            result = -1
    # Pull out the factors of 2 in n, using the (a | 2) supplement.
    if n % 2 == 0:
        if a % 2 == 0:
            return 0
        e = 0
        while n % 2 == 0:
            n //= 2
            e += 1
        if e % 2 == 1 and a % 8 not in (1, 7):
            result = -result
    # n is now odd and positive: Jacobi symbol (a | n) by quadratic reciprocity.
    a %= n
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def is_fundamental_discriminant(d: int) -> bool:
    """True if ``d`` is a fundamental discriminant (``d != 1``).

    ``d`` is fundamental iff either ``d == 1 (mod 4)`` and ``d`` is squarefree, or
    ``d == 0 (mod 4)`` with ``m = d // 4`` squarefree and ``m == 2 or 3 (mod 4)``.
    These are exactly the discriminants of quadratic fields; each carries a real
    primitive character of conductor ``|d|``.
    """
    if d == 0 or d == 1:
        return False
    if d % 4 == 1:
        return _is_squarefree(d)
    if d % 4 == 0:
        m = d // 4
        return (m % 4 in (2, 3)) and _is_squarefree(m)
    return False


def _is_squarefree(m: int) -> bool:
    m = abs(m)
    if m == 0:
        return False
    d = 2
    while d * d <= m:
        if m % (d * d) == 0:
            return False
        if m % d == 0:
            m //= d
        else:
            d += 1
    return True


def fundamental_discriminants(
    d_max: int, *, real: bool = True, imaginary: bool = True
) -> list[int]:
    """Fundamental discriminants ``d`` with ``0 < |d| <= d_max`` (ascending ``|d|``).

    ``real`` keeps ``d > 0`` (real quadratic fields, even characters); ``imaginary``
    keeps ``d < 0`` (imaginary quadratic fields, odd characters). Both subfamilies
    are symplectic; the union is the full quadratic family used by the experiment.
    Ordered by ``|d|`` then sign so the lowest-conductor members come first.
    """
    out: list[int] = []
    for m in range(2, d_max + 1):
        if imaginary and is_fundamental_discriminant(-m):
            out.append(-m)
        if real and is_fundamental_discriminant(m):
            out.append(m)
    return out


def quadratic_character(d: int) -> np.ndarray:
    """Period-``|d|`` table ``chi_d[0..|d|-1]`` of the quadratic character ``(d | .)``.

    For a fundamental discriminant ``d`` the map ``n -> (d | n)`` (Kronecker
    symbol) is the real primitive Dirichlet character of conductor ``|d|``. The
    returned ``complex128`` array is indexed by residue (``chi_d[n % |d|]``), so it
    plugs directly into :func:`dirichlet.lfunction_weights`,
    :func:`dirichlet.lfunction_value`, and the locator -- the only
    number-theoretic input the family statistic consumes.
    """
    if not is_fundamental_discriminant(d):
        raise ValueError(f"{d} is not a fundamental discriminant")
    q = abs(d)
    chi = np.array([kronecker_symbol(d, n) for n in range(q)], dtype=np.complex128)
    return chi


# --- Conductor rescaling ------------------------------------------------------


def conductor_density(conductor: int) -> float:
    """Conductor mean-density factor ``log(q) / (2 pi)`` for conductor ``q``.

    The smooth zero-counting function of ``L(s, chi)`` of conductor ``q`` has local
    density ``(1/2 pi) log(q |t| / (2 pi))`` at height ``t`` (verified by direct zero
    counts against the family members). The Katz-Sarnak rescaling uses the
    *conductor* part of this density, ``log(q) / (2 pi)`` -- height-independent, so
    each member is rescaled by its own conductor regardless of where its zeros fall.
    The remaining ``log(|t| / 2 pi)`` term is what makes the empirical density drift
    above ``1`` in the bulk at finite conductor; it shrinks like ``1 / log q`` as the
    conductor grows, so the symmetry type is read **near the central point**
    ``x -> 0`` (the standard finite-conductor Katz-Sarnak picture). Arithmetic
    conductor ``log(q)`` is used rather than ``log(q/pi)`` so the factor is positive
    for every fundamental discriminant (the smallest, ``q = 3, 4``, fall below
    ``pi``); the constant inside the log does not change the universal kernel shape.
    """
    return float(np.log(conductor) / (2.0 * PI))


def rescale_zeros(gammas: np.ndarray, conductor: int) -> np.ndarray:
    """Rescale ordinates ``gamma`` to ``x = gamma * log(q) / (2 pi)``."""
    return np.asarray(gammas, dtype=np.float64) * conductor_density(conductor)


# --- RMT 1-level density kernels (the forward comparison) ---------------------
#
# numpy's sinc is the normalised sinc, np.sinc(t) = sin(pi t)/(pi t), so
# np.sinc(2 x) = sin(2 pi x)/(2 pi x) -- exactly the kernel term below.


def symplectic_density(x: np.ndarray | float) -> np.ndarray:
    """Symplectic (USp) 1-level density ``W_Sp(x) = 1 - sin(2 pi x)/(2 pi x)``.

    Vanishes at ``x = 0`` (zeros repelled from the central point) -- the predicted
    kernel for the quadratic Dirichlet family.
    """
    x = np.asarray(x, dtype=np.float64)
    return 1.0 - np.sinc(2.0 * x)


def unitary_density(x: np.ndarray | float) -> np.ndarray:
    """Unitary (U) 1-level density ``W_U(x) = 1`` (no central enhancement)."""
    return np.ones_like(np.asarray(x, dtype=np.float64))


def orthogonal_even_density(x: np.ndarray | float) -> np.ndarray:
    """Even-orthogonal (SO even) density ``W_SO+(x) = 1 + sin(2 pi x)/(2 pi x)``.

    Enhanced at ``x = 0`` (``W_SO+(0) = 2``) -- the opposite of symplectic.
    """
    x = np.asarray(x, dtype=np.float64)
    return 1.0 + np.sinc(2.0 * x)


def orthogonal_odd_density(x: np.ndarray | float) -> np.ndarray:
    """Odd-orthogonal (SO odd) density away from the origin: ``1 - sin(2 pi x)/(2 pi x)``.

    Equals the symplectic kernel except for a Dirac mass at ``x = 0`` (the zero
    forced by the odd functional equation), which a binned empirical density cannot
    resolve; provided for completeness, not used in the discrimination.
    """
    x = np.asarray(x, dtype=np.float64)
    return 1.0 - np.sinc(2.0 * x)


SYMMETRY_KERNELS = {
    "symplectic": symplectic_density,
    "unitary": unitary_density,
    "orthogonal_even": orthogonal_even_density,
}


# --- Family aggregation -------------------------------------------------------


def member_rescaled_zeros(
    d: int, *, x_max: float = 4.0, e_min: float = 0.05, dps: int = 15
) -> np.ndarray:
    """Rescaled low-lying zeros ``x = gamma * log(|d|)/(2 pi)`` of ``L(s, chi_d)``.

    Computes the independent mpmath zeros up to the ordinate corresponding to
    ``x_max`` (forward ground truth, :func:`dirichlet.lfunction_zeros`) and rescales
    them by the conductor. The zeros are produced to be compared, never fed back.
    """
    q = abs(d)
    e_max = float(x_max) / conductor_density(q)
    gammas = lfunction_zeros(quadratic_character(d), e_min=e_min, e_max=e_max, dps=dps)
    x = rescale_zeros(gammas, q)
    return x[x <= x_max]


def family_rescaled_zeros(
    discriminants: list[int], *, x_max: float = 4.0, e_min: float = 0.05, dps: int = 15
) -> tuple[np.ndarray, int]:
    """Pool the rescaled low-lying zeros over a family of discriminants.

    Returns ``(rescaled_zeros, n_members)`` -- the concatenated rescaled ordinates
    of every member and the member count, the two inputs
    :func:`one_level_density` needs to normalise the empirical kernel.
    """
    pooled: list[np.ndarray] = []
    for d in discriminants:
        pooled.append(member_rescaled_zeros(d, x_max=x_max, e_min=e_min, dps=dps))
    rescaled = np.concatenate(pooled) if pooled else np.empty(0)
    return rescaled, len(discriminants)


def family_one_level_density(
    discriminants: list[int],
    *,
    x_max: float = 3.0,
    n_bins: int = 12,
    e_min: float = 0.05,
    dps: int = 15,
) -> tuple[np.ndarray, np.ndarray, int]:
    """End-to-end family 1-level density: discriminants -> ``(centres, W, n_members)``.

    The one-call entry point the script, plot, and test share: pools the rescaled
    low-lying zeros over the family (:func:`family_rescaled_zeros`) and bins them
    into the empirical kernel (:func:`one_level_density`). Forward throughout -- the
    zeros are computed independently per member and only ever compared.
    """
    rescaled, n_members = family_rescaled_zeros(
        discriminants, x_max=x_max, e_min=e_min, dps=dps
    )
    centres, w = one_level_density(rescaled, n_members, x_max=x_max, n_bins=n_bins)
    return centres, w, n_members


def one_level_density(
    rescaled: np.ndarray, n_members: int, *, x_max: float = 4.0, n_bins: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical family 1-level density ``W(x)`` from pooled rescaled zeros.

    The 1-level density is defined so the expected number of rescaled zeros in
    ``[x, x + dx]``, averaged over the family, is ``W(x) dx``; the empirical
    estimate is therefore ``counts / (n_members * bin_width)`` (so ``W -> 1`` in the
    bulk, matching the kernels). Returns ``(centres, W)``.
    """
    rescaled = np.asarray(rescaled, dtype=np.float64)
    counts, edges = np.histogram(rescaled, bins=n_bins, range=(0.0, x_max))
    width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = counts / (n_members * width) if n_members else np.zeros_like(counts, float)
    return centres, w


def symmetry_distances(
    centres: np.ndarray, density: np.ndarray, *, x_disc: float = 1.5
) -> dict[str, float]:
    """RMS distance from the empirical ``W(x)`` to each symmetry kernel near 0.

    Restricts to ``x <= x_disc`` -- the region where the three kernels differ most
    (the central enhancement/suppression) and where the symmetry type is decided.
    Returns ``{name: rms}`` for each kernel in :data:`SYMMETRY_KERNELS`.
    """
    centres = np.asarray(centres, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    mask = centres <= x_disc
    out: dict[str, float] = {}
    for name, kernel in SYMMETRY_KERNELS.items():
        diff = density[mask] - kernel(centres[mask])
        out[name] = float(np.sqrt(np.mean(diff**2)))
    return out


def classify_symmetry(
    centres: np.ndarray, density: np.ndarray, *, x_disc: float = 1.0
) -> str:
    """Name of the closest symmetry kernel (the forward verdict)."""
    distances = symmetry_distances(centres, density, x_disc=x_disc)
    return min(distances, key=distances.get)


# --- GPU locator at family scale (the forward producer) -----------------------


def locate_member_zeros(
    d: int,
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, str]:
    """Forward-locate the zeros of ``L(s, chi_d)`` on ``grid`` via the locator.

    Reuses the prime-driven mirror locator (:mod:`dirichlet_locator`, GPU NVRTC with
    a CPU fallback) on the quadratic character ``chi_d``: ``|M'_z(E)|`` peaks at the
    ordinates of the zeros. Returns ``(located_ordinates, backend)``. This is the
    *forward producer* that scales over a family -- each member is an independent
    locator scan, embarrassingly parallel on the GPU -- whereas the precise
    near-central statistic uses the mpmath ground truth (:func:`member_rescaled_zeros`);
    the two are cross-checked in the tests. No zero enters the scan.
    """
    from .dirichlet_locator import local_maxima, peak_threshold, scan_locator

    char = quadratic_character(d)
    values, backend = scan_locator(char, n, grid, sigma=sigma, prefer_gpu=prefer_gpu)
    peaks = np.asarray(grid, dtype=np.float64)[
        local_maxima(np.abs(values), peak_threshold(n))
    ]
    return peaks, backend


def locate_family_zeros(
    discriminants: list[int],
    n: int,
    grid: np.ndarray,
    *,
    sigma: float = 0.5,
    prefer_gpu: bool = True,
) -> tuple[list[np.ndarray], str]:
    """Batched forward-locate the whole family's zeros in **one** GPU launch (#68).

    The Phase-2 batched generalisation of :func:`locate_member_zeros`: builds every
    member's quadratic character, scans ``|M'_z(E)|`` for the *entire family* across
    the whole ``grid`` at once via the batched kernel
    (:mod:`dirichlet_locator_family_gpu`, GPU NVRTC with a CPU fallback over
    :func:`dirichlet_locator_family.family_scan`), then reads the peaks per member.
    Returns ``(per_member_peaks, backend)`` -- ``per_member_peaks[i]`` are the located
    ordinates of ``L(s, chi_{discriminants[i]})``.

    This is the embarrassingly-parallel forward producer at family scale -- the GPU
    "scale" leverage #51 called out -- and it consumes no zero (they are read *off*
    ``|M'_z|``). The precise near-central statistic still uses the mpmath ground
    truth (:func:`member_rescaled_zeros`); the two are cross-checked in the tests.
    """
    from . import dirichlet_locator_family as dlf

    chars = [quadratic_character(d) for d in discriminants]
    grid = np.asarray(grid, dtype=np.float64)
    if prefer_gpu:
        try:
            from . import dirichlet_locator_family_gpu as gpu

            block = gpu.family_scan_gpu(chars, n, grid, sigma=sigma)
            return dlf.locate_family_peaks(np.abs(block), grid, n), "gpu"
        except ImportError:
            pass  # cupy missing -> CPU reference below
    block = dlf.family_scan(chars, n, grid, sigma=sigma)
    return dlf.locate_family_peaks(np.abs(block), grid, n), "cpu"
