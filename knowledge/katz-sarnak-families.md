# Katz–Sarnak family statistics — the quadratic Dirichlet family is symplectic (#51)

*Why* a whole **family** of `L`-functions belongs here, what its low-lying zeros
do, and how the CPU ground truth and GPU locator split the work. Read
[`project-framing.md`](project-framing.md) first for the forward-vs-inverse rule;
this is the family-statistics companion to the single-sequence GUE warm-up
([`../README.md`](../README.md) and the `spacing` module) and reuses the
Dirichlet-`L` machinery of the Dirac-mirror track ([`dirac-mirror.md`](dirac-mirror.md)).

## Why this is forward, not inverse

The discriminating test is *"does it consume the zeros as input?"* No. Each
member's zeros are produced **independently** — from its character alone
(`mpmath`, the analogue of `zeros.py`) — and the random-matrix symmetry kernel is
a **parameter-free prediction** we compare the pooled density against. The only
input is the discriminant `d` (pure number theory: the quadratic character
`χ_d(n) = (d | n)`, a Kronecker symbol). Nothing is fit. Litmus: corrupt a
member's zeros and the family verdict degrades toward noise — it does not "track"
a planted answer, because there is no inverse step to track it.

## The object: Katz–Sarnak universality over a family

Montgomery–Odlyzko says the *single* sequence of `ζ` zeros follows the **unitary**
(GUE) ensemble. Katz–Sarnak is the family-level refinement: for a natural family
of `L`-functions, the low-lying zeros — those near the central point `s = 1/2`,
rescaled by the conductor — follow the **1-level density** of one of the classical
compact groups, and *which* group is fixed by the family's symmetry type.

The cleanest case is the family of **quadratic Dirichlet `L`-functions**
`L(s, χ_d)` over fundamental discriminants `d`. Its symmetry type is **symplectic**
(Katz–Sarnak; Özlük–Snyder; Rubinstein 2001), so the family-averaged 1-level
density should track

    W_Sp(x) = 1 − sin(2πx) / (2πx),

which **vanishes at the central point** `x = 0` — zeros are *repelled* from the
center. The contrast kernels are unitary `W_U(x) = 1` (flat) and even-orthogonal
`W_SO+(x) = 1 + sin(2πx)/(2πx)` (enhanced, `W_SO+(0) = 2`). The central
suppression/enhancement is the whole discriminating signal.

## The character: a Kronecker symbol

The quadratic character of a fundamental discriminant `d` is `n ↦ (d | n)`, the
Kronecker symbol — a real primitive Dirichlet character of conductor `|d|`. Built
directly (`katz_sarnak.quadratic_character`, via `kronecker_symbol`) rather than
through a primitive root, it yields the **whole family** in one elementary
routine: every fundamental discriminant, real (`d > 0`, even character) and
imaginary (`d < 0`, odd character). It is just a period-`|d|` array, so it drops
straight into the existing `dirichlet` API (`lfunction_zeros`, the locator) — no
general composite-character machinery needed. Cross-checked against the existing
characters: `χ_5 = (5|·)` equals the Legendre symbol `(·|5)`, and `χ_{−4}` is the
Dirichlet-beta character mod 4.

## The conductor rescaling (and the finite-conductor caveat)

A direct zero-count of the family members confirms the smooth density of `L(s, χ)`
of conductor `q` at height `t` is

    ρ(t) = (1/2π) log( q |t| / (2π) )

(the `ζ` Riemann–von Mangoldt density with `q` inside the log; verified, not
assumed). Katz–Sarnak rescales each member's ordinates by the **conductor** part,
height-independent:

    x = γ · log(q) / (2π)          (arithmetic conductor; positive for all q ≥ 3)

so each member is scaled by its own conductor regardless of where its zeros fall.
The remaining `log(|t|/2π)` term is what makes the empirical density drift **above
1 in the bulk** at finite conductor; it shrinks like `1/log q` as the conductor
grows. So — exactly as in the standard finite-conductor Katz–Sarnak numerics — the
symmetry type is read **near the central point** `x → 0`, not from the bulk.
(`log(q)` is used rather than `log(q/π)` only so the factor stays positive for the
smallest discriminants `q = 3, 4 < π`; the constant inside the log does not change
the universal kernel shape.)

## The forward result

Pooling the independently-computed low-lying zeros over the quadratic family
(`|d| ≤ 48`, ~30 members) and binning the conductor-rescaled `x` gives an empirical
1-level density that is **decisively symplectic**: the lowest bins sit at ≈ 0
(strong central suppression), and the symplectic kernel is the closest of the
three near the center (RMS distance on `x ≤ 1`: symplectic `< unitary <
orthogonal`, a comfortable margin), while unitary is flat at 1 and orthogonal is
enhanced to ≈ 2 there. The bulk mean sits a little below the kernels' 1 at this
modest conductor — the finite-conductor drift above, read accordingly. The
suppression is the symplectic signature, and it is produced by the number theory,
never fit.

## The numerics: CPU ground truth, GPU for scale

Per the house rule, the CPU reference is the truth:

- **`katz_sarnak`** (CPU) — the Kronecker character, fundamental-discriminant
  enumeration, conductor rescaling, the three RMT kernels, and the family
  1-level-density aggregation + symmetry discrimination. The precise near-central
  statistic uses the **mpmath** ground-truth zeros (`dirichlet.lfunction_zeros`)
  because the central region demands accurate ordinates.
- **GPU** — the family is *embarrassingly parallel*: each member is one independent
  locator scan. `katz_sarnak.locate_member_zeros` runs the prime-driven mirror
  locator ([`dirac-mirror.md`](dirac-mirror.md), `dirichlet_locator`, GPU NVRTC
  with a CPU fallback) on `χ_d` as the **forward producer at scale**, cross-checked
  against the mpmath zeros. This is where the GPU earns its place — many `L`-functions
  in parallel — in contrast to the single high-precision flagship eigensolve. A
  dedicated batched-family GPU kernel (one scan grid across the whole family at
  once) is the natural Phase-2 follow-up.

## Reproduce

    uv run python scripts/run_katz_sarnak.py                  # family 1-level density + verdict
    uv run python scripts/run_katz_sarnak.py --d-max 80 --plot # larger family, save the figure
    uv run python scripts/run_katz_sarnak.py --d-max 60 --locate # GPU locator over the family

The forward family-density test is marked `slow` (`uv run pytest -m slow` to
include it); the character algebra, kernels, and discrimination logic run in the
default suite.

## Sources

- **N. Katz & P. Sarnak**, *Random Matrices, Frobenius Eigenvalues, and Monodromy*
  (AMS, 1999), and *Zeroes of zeta functions and symmetry*, Bull. AMS **36**
  (1999). The symmetry-type framework and the 1-level density kernels.
- **M. Rubinstein**, *Low-lying zeros of `L`-functions and random matrix theory*,
  Duke Math. J. **109** (2001). The quadratic Dirichlet family computed against the
  symplectic prediction — the numerical template this experiment follows.
- **A. Özlük & C. Snyder** (1999); **Iwaniec, Luo & Sarnak**, *Low lying zeros of
  families of `L`-functions*, Publ. IHÉS **91** (2000). The symplectic symmetry of
  the quadratic family and the conductor-rescaled 1-level density.
- **arXiv:2509.05810** (Dillon, Miller, et al., 2025) — recent `n`-level-density
  *analytic* results for GL(2) families; the open **numerical** companion at scale
  is the niche this experiment occupies (see
  [`frontier-survey-2026.md`](frontier-survey-2026.md), candidate **F1**).
