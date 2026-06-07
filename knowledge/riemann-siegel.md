# GPU Riemann–Siegel ζ-evaluator — a forward tool (#55)

*What* the Riemann–Siegel evaluator computes, *why* it is a forward-neutral tool
(not an experiment), where the GPU win is (and the fp64 height ceiling), and what
it unlocks downstream. Read [`project-framing.md`](project-framing.md) first.

## Forward? — neutral, it's a tool

This evaluates `ζ(1/2 + i t)` (and the Hardy function `Z`) **from the
Riemann–Siegel expansion** — a structural identity, no zeros consumed. It is a
*tool*, like a faster `mpmath.zeta`: whatever is built on top of it (zero
verification, value distribution, De Bruijn–Newman at height) must itself stay
forward. The tests pin every piece against `mpmath` (`siegeltheta` / `siegelz` /
`zeta`), the house ground-truth rule.

## The object

With `τ = √(t/2π)`, `N = ⌊τ⌋`, `p = τ − N`, and `θ` the Riemann–Siegel theta
function,

    Z(t) = 2 Σ_{n=1}^{N} cos(θ(t) − t log n)/√n
           + (−1)^{N−1} τ^{−1/2} [ C₀(p) + C₁(p) τ^{−1} + C₂(p) τ^{−2} ],

and `ζ(1/2 + i t) = Z(t) e^{−i θ(t)}` (`Z` is real for real `t`, so the on-line
zeros of `ζ` are the real zeros of `Z`). The cost is `O(√t)` per height, versus
`O(t)` for a naive Dirichlet sum — the whole point of the formula.

The correction coefficients come from the single function
`Ψ(p) = cos(2π(p² − p − 1/16)) / cos(2π p)` (Gabcke / Edwards Ch. 7):

    C₀ = Ψ,   C₁ = −Ψ‴ / (96 π²),   C₂ = Ψ⁽⁶⁾/(18432 π⁴) + Ψ″/(64 π²).

These three terms reach ~1e-6 by `t = 50` and ~1e-11 by `t = 1e5` — the expansion
is **asymptotic**, so it sharpens with height. (A fourth term `C₃` was tried and
dropped: it degraded the result at every height we tested below `t ≈ 1e5` — the
standard `Ψ⁽⁹⁾` coefficient is both wrong as we transcribed it and numerically
hostile, and two terms already cover the useful range.)

## Two precision facts (don't forget)

1. **`Ψ` is entire — but the naive derivative recurrence is not.** Numerator and
   denominator of `Ψ` share simple zeros at `p ∈ {1/4, 3/4}`, so those poles are
   *removable* and `Ψ` is analytic everywhere. A quotient/Leibniz recurrence for
   its derivatives divides by `cos(2π p) → 0` once per order, which detonates near
   those points (we saw `Z` spike to ~2e4 at `t ≈ 113.5`, where `τ ≈ 4.25`). The
   fix is a **Cauchy integral** for each `Ψ⁽ᵏ⁾` on a complex circle whose two
   real-axis crossings dodge `1/4, 3/4`; `Ψ` is then `O(1)` everywhere on the
   contour and the trapezoidal rule is spectrally accurate (matches `mpmath` to
   ~1e-10 through order 6, even *at* `p = 1/4`). All fp64, no `mpmath` in the hot
   path — which is what keeps the GPU path pure fp64.

2. **fp64 has a height ceiling, set by the phase argument, not the magnitude.**
   `Z` itself is an `O(1)`, cancellation-free object (unlike the flagship
   eigensolve or the De Bruijn–Newman integral), so fp64 is the *right* tool — but
   the main-sum phase `t log n` is `O(t)`, and at `t ~ 1e8` its fp64 ULP is already
   ~2e-7, capping `Z` to ~6–7 digits there (GPU and CPU agree to that same level —
   they compute the same thing). This is the "precision budget per height" #20
   flagged. Beyond it you need an extended-precision phase (or a different
   algorithm); within it fp64 is a clean win.

## Where the GPU wins

The main sum is the embarrassingly-parallel piece: one CUDA thread per height,
looping over its own `N` terms (`kernels/riemann_siegel.cu`), versus the numpy
reference's per-`n` masked accumulation. The `θ` phase and the asymptotic
remainder are `O(n_heights)` host work, shared by both paths. Measured on the
RTX 3090 (main sum, 20–400k heights): **~9×** at `t = 1e4` (`N ≈ 39`), **~52–83×**
at `t = 1e6` (`N ≈ 400`), **~131×** at `t = 1e8` (`N ≈ 4000`) — the speedup grows
with `N`, i.e. with height. (Below `t ≈ 1.5e4`, `N < 48` and the shared remainder
dominates, so there is no full-`Z` win there; the kernel still matches the CPU to
fp tolerance.)

## What it unlocks

- **De Bruijn–Newman at height (#20).** The #20 spike deferred pushing `H_t` to
  real height because the naive forward integral is precision-bound; on-line `ζ`
  via Riemann–Siegel is the precision-friendly route to the `t = 0` zeros at
  height (and the foundation for an effective-`H_t` expansion).
- **Fast independent zero verification** for the Dirichlet-`L` locator (#60) and
  the Katz–Sarnak family work (#51): `Z` sign changes bracket the on-line zeros
  with no `mpmath.zetazero` calls.
- **Value-distribution studies** of `ζ` on the critical line (a future forward
  experiment — the tool is neutral; the experiment must stay forward).

**Deferred:** the Odlyzko–Schönhage many-point amortization (a band-limited
Taylor/FFT scheme that shares the main sum across nearby heights). Per-height GPU
parallelism already serves the current consumers; OS is the next lever for very
large at-once evaluations.

## Reproduce

    uv run python scripts/run_riemann_siegel.py        # GPU scan, asserts CPU match, locates zeros

## Sources

- H. M. Edwards, *Riemann's Zeta Function*, Ch. 7 — the Riemann–Siegel formula and
  the remainder coefficients `C_k` in terms of `Ψ` and its derivatives.
- W. Gabcke, *Neue Herleitung und explizite Restabschätzung der
  Riemann–Siegel-Formel* (1979) — rigorous remainder bounds / coefficients.
- A. M. Odlyzko & A. Schönhage, *Fast algorithms for multiple evaluations of the
  Riemann zeta function*, Trans. AMS **309** (1988) — the deferred many-point
  scheme.
