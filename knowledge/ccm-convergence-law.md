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

## The gain law: convergence tracks the log-window, not the primes (#94)

What *governs* that super-exponential low-zero rate, step by step? Groskin raised a
sharp forward observation on the `connes-cvs` reproduction thread
([akivag613/connes-cvs-#1](https://github.com/akivag613/connes-cvs-/issues/1)): the
largest single-step gain in his sweep involves **no new prime** — from `c=13` to
`c=14`, `π(14)=π(13)=6`, yet the first-zero error falls nearly six orders of
magnitude — and across the sweep the per-step gain correlates with the interval
length `L = log c` (`r ≈ −0.96`), **not** with prime content.

Because our operator is built forward from the primes, this is directly checkable on
an independent assembly. [`first_zero_gain_law`](../src/zeta_spectral_gpu/ccm_convergence.py)
sweeps the first-zero error over integer cutoffs and correlates the per-step
order-of-magnitude gain `log10(err₀/err₁)` against the log-window (`ln c`, `Δ ln c`)
versus the arithmetic content (`Δπ(c)`, `Δ#prime-powers`). At `N = 80`:

| step | gain (orders) | `Δ ln c` | `Δπ` | `Δ#pp` |
|---|---|---|---|---|
| 11→12 | 5.31 | 0.087 | **0** | **0** |
| 12→13 | 5.12 | 0.080 | 1 | 1 |
| 13→14 | 5.30 | 0.074 | **0** | **0** |
| 14→15 | 5.26 | 0.069 | **0** | **0** |
| 16→17 | 4.60 | 0.061 | 1 | 1 |

The two largest steps in the window (`11→12`, `13→14`) add **no new prime or prime
power**. Per-step gain vs prime content is essentially uncorrelated (`r ≈ −0.05` vs
`Δπ`, `−0.08` vs `Δ#pp`); vs the log-window it is strong and negative (`r ≈ −0.74`
at `N=80, c=10..18`, tightening to `≈ −0.98` at `N=48, c=11..16` where the finite-`N`
floor makes the gains cleanly monotone). The exact magnitude is `N`/window-sensitive,
but the ordering is not: **the log-window `2 ln λ = ln c` governs the gain, the prime
count does not.** This is the empirical fingerprint of the log-window structure the
operator is built on — reproduced forward, independent of `connes-cvs`. Reproduce:
`scripts/run_ccm_convergence.py --mode gain-law`.

**The `c=100` datapoint (Groskin's binding-constraint cell).** An independent
first-zero assembly at `c=100` is cheap: `|γ₁ − t₁| = 7.30e-211` at `N=120, dps=520`
in ~15 s, **identical at `dps=640`** — so the cell is **finite-`N`-limited, not
precision-limited** (the binding constraint is the `dps ≈ 500` Groskin flags, but
once met the error floor is set by `N`, not the digits). Pushing to `N=160` deepens
it to `1.78e-253`. Since the first-zero error at fixed `c` is a function of
`(N, T, dps)`, a digit-for-digit diff against `connes-cvs` needs Groskin's `c=100`
cell parameters (requested on the thread); our assembly's archimedean term is
closed-form (no `T`-quadrature), so matching needs only his `N` and adequate `dps`.

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

   The fp64 "error" is **entirely** `ξ`-corruption — *over the low zeros*. This is
   the regime where Groskin's super-exponential convergence lives, so a fp64
   "measurement" there is the precision wall, not the finite-cutoff law. **This does
   *not* extend to Śliwiński's uniform-error Conjecture 4.1**, which is edge-dominated
   — see the #82 correction below. *(Theorem 3.1 is rigorous and precision-independent
   — untouched either way.)*

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

## #82: the edge is robust — the "Conj 4.1 ≈ precision wall" caveat overclaims

The Phase-0 corruption finding above is measured on the **low** zeros. Conjecture
4.1, though, is about the **uniform** error `E = max_k |ν_k − ζ_k|`, which is
**edge-dominated**: the largest error sits at the resolution edge `k → N`, where the
`N`-th eigenvalue is pinned to the window-edge pole `d_N = 2πN/L`. Does fp64
`ξ`-corruption reach *there*? If the edge is robust, the caveat evaporates.

[`ccm_convergence.edge_corruption_profile`](../src/zeta_spectral_gpu/ccm_convergence.py)
answers it forward: compute the genuine (mpmath) and the fp64-`ξ` spectra through the
*same* secular root-finder (only `ξ` differs), compare both to the zeros
index-by-index, and tag each root by its pole gap. At `x = 13`, `dps = 88`:

- **The corruption is confined to the low / near-null band and decays toward the
  edge.** Near the edge the fp64 and mpmath roots land in the *same* pole gap — both
  pinned to the bulk `d_n` — so `|ν_k^{fp64} − ν_k^{mpmath}|` falls to `~1` there,
  while in the low band it is `~10–20`. The edge is the diagonal `diag(2πn/L)`, *not*
  the `ξ` direction fp64 destroys — exactly as the structure predicts.
- **The low-band corruption is bounded in `N`; the genuine edge error grows `~ζ_N`.**
  So there is a crossover `N` above which fp64's uniform error `E` is set by the
  genuine (robust) edge, not the corrupted low band:

  | `N` (`x = 13`) | genuine edge error | low-band corruption | fp64's `E` set by |
  |---|---|---|---|
  | 80  | `6.0`  | `13.7` | corrupted **low band** |
  | 120 | `30.1` | `19.9` | genuine **edge** (`k = 115`) |
  | 160 | `58.1` | `12.6` | genuine **edge** (`k = 152`) |

  The crossover is already at `N ≈ 120` — fully high-precision-reachable. Śliwiński's
  `κ = N = λ ~ 7050` sits *far* above it, where the genuine edge error (`~ζ_N`,
  thousands) dwarfs the bounded `ξ`-corruption (`O(10)`).

**Verdict (NO-GO).** A fp64 measurement of the *uniform* error `E` is edge-dominated
at Śliwiński's scale, and the edge is robust to `ξ`-corruption. So Conjecture 4.1's
fp64 numerics are **plausibly genuine**, not a precision artifact. The
"Conj 4.1 ≈ precision wall" framing **overclaims**: the precision wall is real for
the **low** zeros (Groskin's super-exponential regime — the Phase-0 table above), but
*not* for Śliwiński's uniform-error law. Theorem 3.1 is untouched (rigorous,
precision-independent). No Śliwiński outreach is warranted on the strength of the
caveat. Reproduce: `scripts/run_ccm_convergence.py --mode edge-corruption`.

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
- **Bonus, then corrected (#82):** the precision-artifact caveat applies to the
  **low** zeros, *not* to Śliwiński's uniform-error Conjecture 4.1 — its `E` is
  edge-dominated and the edge is robust to `ξ`-corruption, so the fp64 measurement is
  plausibly genuine. **NO-GO** on any outreach. See "#82: the edge is robust" above.

What is *not* reachable here: Śliwiński's `κ ∼ 7000` scale (fp64-only; fp64 corrupts
the **low** spectrum, though the edge that dominates `E` is pole-pinned and robust,
#82) and a clean confirmation of `lim E·ln λ = 1` (needs that scale). Documented
rather than faked.

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
