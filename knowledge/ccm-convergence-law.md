# The CCM convergence law — measured, and a precision caveat (#65)

> A forward study of *how* the Connes–Consani–Moscovici operator spectrum
> approaches the zeta zeros as the cutoff grows, following Śliwiński
> (arXiv:2601.12133). Built on the repo's proven multiprecision spectrum
> ([`ccm.py`](../src/zeta_spectral_gpu/ccm.py), [`ccm-operator.md`](ccm-operator.md));
> the new layer is [`ccm_convergence.py`](../src/zeta_spectral_gpu/ccm_convergence.py)
> + the generic accelerators in
> [`acceleration.py`](../src/zeta_spectral_gpu/acceleration.py).
>
> **Forward, not inverse.** The spectrum is the prime-built operator's; the zeros
> enter *only* as the yardstick the error is measured against, and *only* after
> the fact. The accelerators consume the forward eigenvalue sequence and never see
> a zero. See [`project-framing.md`](project-framing.md).

## What Śliwiński states

For the operator `D_log^{(λ,N)}` with `2N+1` eigenvalues `ν_k` and the zeros
`ζ_k`, two error notions (Defs 2.5/2.6):

- **Mean absolute error** `ε(λ,N) = (1/n) Σ_{k≤n} |ν_k − ζ_k|`,
- **Uniform error** `E(λ,N) = max_k |ν_k − ζ_k|`.

Two claims:

- **Theorem 3.1 (proven):** `ε(λ,N) ≥ 1/(4 ln λ)`. A Heisenberg/uncertainty
  bound — the operator lives in a log-window of width `2 ln λ`, capping the
  position spread, so the spectral spread (hence the average eigenvalue-to-zero
  distance) is bounded below. In the repo's cutoff `x = λ²` this is `1/(2 ln x)`.
  Corollary 3.2 turns it into `ε(N) = Ω(1/ln N)` under the coupling `N ln λ ∼ p_N`.
- **Conjecture 4.1 (numerical):** `E(λ,N)` is *also* inverse-logarithmic,
  `lim E·ln λ` exists (possibly `= 1`). The evidence is a **fp64 (~7-digit)** GPU
  sweep with `κ = N = λ` up to `7050`, against the first 1000 zeros.

## The reconciliation (with Groskin) that shapes everything

There is an apparent tension. Groskin (arXiv:2605.20224) drives the *first* zero's
error to `1.5e-168` by pushing the cutoff at huge precision — the **low** zeros
converge **super-exponentially**. The repo's own §6 reproduction shows it cleanly:
first-zero error `3.4e-50 → 2.4e-55 → 1.1e-60` at `x = 12,13,14`. That is *not*
inverse-logarithmic.

The resolution: **the inverse-log behaviour is strictly the aggregate /
resolution-edge statement.** The `N`-th eigenvalue always sits at the window edge
`d_N = 2πN/L`; resolving the *top* of the first-`N` set is what runs out at rate
`1/L`. The *low* zeros are nailed (super-exp); the *mean over the full first-`N`*
is forced up to `≥ 1/(4 ln λ)` by the unresolvable edge. They measure different
things.

## What this repo found (Phase-0 of #65)

Resolving the spectrum needs the near-null eigenvector `ξ` of the Weil form, which
lives at the sub-`ε_N` scale. So the precision needed **grows with the cutoff
depth**:

| precision | resolves cutoff up to | source |
|---|---|---|
| fp64 (`~16` digits) | `x ≈ 5–9` | this repo (`ccm_gpu` wall) |
| `dps = 110` | `x ≈ 30–50` | measured here |
| `dps = 500–1000` | `x = 100` (`c = 100`) | Groskin |

Two load-bearing consequences:

1. **fp64 cannot recover the spectrum — only `ε_N`'s conditioning.** The fp64
   near-null eigenvector is roundoff once `ε_N` underflows (which is almost
   immediately), so the fp64 *spectrum* is corrupted, not merely imprecise. And
   `D_log` does not escape this: it is `diag(2πn/L) − |D_log ξ⟩⟨δ_N|`
   (Theorem 1.1), a rank-one update that still needs `ξ`. Confirmed in two regimes;
   the GPU's honest role here is the fp64 **assembly** (already validated to
   `~1e-12` in #9), not the eigensolve.

2. **A fp64 inverse-log "measurement" is largely measuring the wall.** Over a
   fixed low set, the *genuine* (mpmath) error collapses super-exponentially while
   the fp64 error stays `O(10)` — it is `ξ`-corruption, not finite-cutoff error.
   Measured by `fp64_spectrum_corruption`, `N = 80`, first 12 zeros:

   | `x` | genuine max error (mpmath) | fp64 max error | corruption `|ν_fp64 − ν_mpmath|` |
   |---|---|---|---|
   | 11 | `1.8e-24` | `1.1e+01` | `1.1e+01` |
   | 13 | `3.9e-34` | `1.4e+01` | `1.4e+01` |
   | 15 | `4.0e-44` | `2.4e+01` | `2.4e+01` |

   The fp64 "error" is **entirely** `ξ`-corruption. So Śliwiński's Conjecture-4.1
   numerics (run at ~7 digits) cannot be separating the genuine inverse-log law
   from the precision wall; a high-precision check can. *(Theorem 3.1 is rigorous
   and precision-independent — this caveat is only about the numerical Conj 4.1.)*

## The edge picture (Theorem 3.1, high precision)

`ccm_convergence.convergence_errors` + the `edge` study give the per-index error
profile. At `N = 120, x = 13, dps = 88` (tracking 116 zeros):

- first-zero error `2.44e-55` (matches §6 exactly), edge error `≈ 30`;
- the Heisenberg floor `1/(4 ln λ) = 0.195`; the per-index error first **reaches the
  floor at `k = 54`** — the resolution edge — then blows up.

This single-`x` edge index is generalised across the cutoff in #53: the
zero-tracking range `k*(x) = #{k : |ν_k − ζ_k| / ζ_k < 10⁻³}` grows **linearly** in
the prime cutoff (`t* = ζ_{k*} ≈ 12x`), and at `x = 13` it lands at `k* = 53` —
the same detachment this floor crossing marks. See
[`ccm-universality.md`](ccm-universality.md).

So the floor is an **edge phenomenon**: the tracked low zeros sit far below it
(super-exp), and the mean is pushed above it only by the unresolvable top. (We do
not chase the exact full-`N` edge roots: near the edge the secular roots cluster
picket-like and the root-finder — built for the clean low zeros — is unreliable
there. The constant `lim E·ln λ = 1?` therefore needs scale beyond high-precision
reach, exactly why Śliwiński went to fp64 at `κ = 7500`.)

## Does acceleration help? (F2 — the negative result)

`acceleration.py` provides Aitken/Shanks, Wynn-ε, and Richardson/Neville
extrapolation (all arithmetic-generic, so the high-precision `mpf` sequences keep
their digits; all structurally forward — input is a numeric sequence, never a
zero). Applied to the cutoff-sequence `ν_k(x_j)` of a fixed zero:

**It does not sharpen the estimates.** Over `x = 11..14`, for indices `k = 10..40`,
no accelerator beat the raw deepest cutoff (gains `< 1`). The reason is structural:
where the convergence is clean (the low/resolved zeros) it is **super-exponential**,
faster than any geometric accelerator can exploit — the raw deepest cutoff is
already optimal; the only *slow* (inverse-log) part is the unresolvable resolution
edge. Either outcome was informative (issue #65); this one says the "exploit the
law" idea does not pay off for the reachable spectrum — the leverage Groskin/the
flagship already get is from raw precision, not extrapolation.

## Status of the deliverables

- (a) **Confirm/bound the law:** the proven floor `ε ≥ 1/(4 ln λ)` holds; the
  genuine low-zero convergence is super-exponential (not inverse-log); the
  inverse-log behaviour is the resolution edge. **Done, high precision.**
- (b) **Does acceleration sharpen the `c → ∞` estimates:** **no**, and the reason
  is the super-exponential low-zero convergence. **Done (negative).**
- **Bonus:** a precision-artifact caveat on Śliwiński's numerical Conjecture 4.1.

What is *not* reachable here: Śliwiński's `κ ∼ 7000` scale (fp64-only, and fp64
corrupts the spectrum) and a clean confirmation of `lim E·ln λ = 1` (needs that
scale). Documented rather than faked.

## Sources

- **arXiv:2601.12133** — Śliwiński, *Spectral Analysis of the `D_log^(λ,N)`
  Operators* (Jan 2026). The two error notions, Theorem 3.1, Conjecture 4.1.
- **arXiv:2605.20224** — Groskin, *High-Precision Approximation of Riemann Zeros
  via the Truncated Weil Form* (May 2026). The super-exponential low-zero regime;
  the `connes-cvs` oracle.
- **arXiv:2511.22755** — Connes, Consani, Moscovici, *Zeta Spectral Triples*. The
  operator; see [`ccm-operator.md`](ccm-operator.md).
- Context: [`frontier-survey-2026.md`](frontier-survey-2026.md) (candidates F2/F4),
  [`connes-cvs-oracle.md`](connes-cvs-oracle.md).
