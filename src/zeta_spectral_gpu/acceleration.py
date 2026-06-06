"""Sequence-acceleration primitives (issue #65, the "exploit" half / F2).

Generic, dependency-light accelerators that turn a slowly converging sequence
``s_0, s_1, ...`` into a sharper estimate of its limit. Used by
:mod:`ccm_convergence` to extrapolate the CCM operator's cutoff-sequence of
eigenvalue estimates toward the ``cutoff -> infinity`` limit, and to estimate the
asymptotic constant of Sliwinski's inverse-log error law.

**Forward by construction.** Every function here consumes *only* a numeric
sequence (or ``(node, value)`` pairs) — never the zeta zeros. They are pure
numerical transforms of whatever forward sequence is handed in; the zeros enter
the #65 experiment only afterwards, as the yardstick the extrapolant is measured
against. (Litmus, per ``knowledge/frontier-survey-2026.md`` §4: corrupt the input
and the transform tracks the *input's* limit, not any external target.)

All routines are arithmetic-generic: they work elementwise on Python ``float`` or
on ``mpmath.mpf`` / ``mpc`` (so the high-precision CCM sequences accelerate
without losing digits). Degenerate stages — a vanishing denominator, fewer than
the minimum number of terms — yield ``None`` rather than a bogus number, so the
caller can fall back to the raw sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional


def _is_tiny(x, scale) -> bool:
    """True if ``|x|`` is negligible relative to ``scale`` (guards 0/0 stages)."""
    return abs(x) <= 1e-300 + 1e-30 * abs(scale)


def aitken(seq: Sequence) -> list:
    """One Aitken ``Delta^2`` (single Shanks) pass; output has ``len(seq) - 2``.

    For a sequence with geometric error ``s_n - s ~ A r^n`` (``|r| < 1``), the
    transform ``t_n = s_n - (s_{n+1}-s_n)^2 / (s_{n+2}-2 s_{n+1}+s_n)`` cancels the
    leading geometric term and converges faster to the same limit ``s``. A
    near-zero second difference (already-flat triple) yields ``None`` in that slot.
    """
    out: list = []
    for n in range(len(seq) - 2):
        d1 = seq[n + 1] - seq[n]
        d2 = seq[n + 2] - 2 * seq[n + 1] + seq[n]
        if _is_tiny(d2, seq[n + 2]):
            out.append(None)
        else:
            out.append(seq[n] - d1 * d1 / d2)
    return out


def shanks(seq: Sequence, *, passes: int = 1) -> list:
    """Iterated Aitken: apply :func:`aitken` ``passes`` times.

    Each pass removes one geometric component and shortens the sequence by two.
    ``None`` slots from a degenerate stage are dropped before the next pass (so a
    single flat triple does not poison the whole tail). Returns the most-accelerated
    surviving sequence (possibly shorter than ``len(seq) - 2*passes``).
    """
    cur = list(seq)
    for _ in range(passes):
        if len(cur) < 3:
            break
        cur = [t for t in aitken(cur) if t is not None]
    return cur


def wynn_epsilon(seq: Sequence) -> Optional[object]:
    """Best limit estimate from Wynn's epsilon-algorithm (model-free Shanks/Pade).

    Builds the epsilon tableau

        eps_{-1}^{(n)} = 0,  eps_0^{(n)} = s_n,
        eps_{k+1}^{(n)} = eps_{k-1}^{(n+1)} + 1 / (eps_k^{(n+1)} - eps_k^{(n)}),

    whose even columns ``eps_{2j}`` are successive Shanks/Pade approximants to the
    limit (the odd columns are auxiliary). Returns the deepest stable even-column
    entry — the most accelerated estimate — or ``None`` if the table collapses
    before any even column past the raw sequence is reached. Robust to several
    superimposed geometric/transient components, which iterated :func:`aitken` is
    not. Reference: Wynn (1956); Brezinski & Redivo-Zaglia.
    """
    n0 = len(seq)
    if n0 < 3:
        return None
    # Column-major tableau: col[k] holds eps_k^{(n)} for the current diagonal.
    prev_prev: list = [0 * seq[0]] * n0  # eps_{-1}: zeros (same type as seq)
    prev: list = list(seq)  # eps_0 = s_n
    best = None
    k = 0
    while len(prev) >= 2:
        cur: list = []
        for n in range(len(prev) - 1):
            denom = prev[n + 1] - prev[n]
            if _is_tiny(denom, prev[n + 1]):
                cur.append(None)
            else:
                # prev_prev is one element shorter than prev; index n aligns to
                # eps_{k-1}^{(n+1)}.
                base = prev_prev[n + 1] if (n + 1) < len(prev_prev) else 0 * seq[0]
                cur.append(base + 1 / denom)
        # An even column (k+1 even) holds genuine limit approximants.
        if (k + 1) % 2 == 0:
            live = [c for c in cur if c is not None]
            if live:
                best = live[-1]
        if all(c is None for c in cur):
            break
        # Replace None with a large sentinel so downstream differences stay finite;
        # in practice the tail is trimmed by the length contraction.
        prev_prev = prev
        prev = [c if c is not None else (0 * seq[0]) for c in cur]
        k += 1
    return best


def neville_extrapolate(nodes: Sequence, values: Sequence, *, at=None) -> object:
    """Polynomial (Richardson/Neville) extrapolation of ``values`` to ``at``.

    Given samples ``value_i = f(node_i)`` of a function with an asymptotic
    expansion in the node variable, builds the Neville tableau and returns the
    interpolating polynomial evaluated at ``at`` (default: ``0 * nodes[0]``, i.e.
    the ``node -> 0`` limit). For the inverse-log law set ``node = 1/ln(lambda)``:
    ``f(h) = c + a h + b h^2 + ...`` and the ``h -> 0`` value is the asymptotic
    constant ``c``. The number of nodes sets the polynomial order, so pass only as
    many as the expansion is trusted to (Richardson over-extrapolation is the usual
    failure mode). Requires ``len(nodes) == len(values) >= 1``.
    """
    if len(nodes) != len(values) or not values:
        raise ValueError("nodes and values must be equal, non-empty lengths")
    target = (0 * nodes[0]) if at is None else at
    # Neville: tableau[i] iteratively becomes P_{i..i+m}(target).
    tableau = list(values)
    x = list(nodes)
    m = len(values)
    for level in range(1, m):
        for i in range(m - level):
            num = (target - x[i + level]) * tableau[i] + (x[i] - target) * tableau[
                i + 1
            ]
            tableau[i] = num / (x[i] - x[i + level])
    return tableau[0]


def richardson_limit(nodes: Sequence, values: Sequence, *, window: int = 0) -> object:
    """The ``node -> 0`` Neville extrapolant, optionally over a trailing window.

    Convenience wrapper around :func:`neville_extrapolate` at ``0``. With
    ``window > 0`` only the last ``window`` (node, value) pairs are used — the
    smallest nodes, where the asymptotic expansion is most accurate and a
    low-order extrapolation is safest. ``window = 0`` uses all samples.
    """
    if window and window < len(nodes):
        nodes = list(nodes)[-window:]
        values = list(values)[-window:]
    return neville_extrapolate(nodes, values)
