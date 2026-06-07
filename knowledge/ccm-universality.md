# ccm-universality.md — the prime-cutoff rigidity trend (#18/#53, building on #9/#35)

*Why* the finite-cutoff CCM operator's local statistics look the way they do as
the prime cutoff grows, and which readout to trust. The operator and its
λ-sweep are specified in [`ccm-operator.md`](ccm-operator.md); this note is the
characterisation of its *spectrum as a point process*. Forward throughout: the
operator is built from the von Mangoldt sum over `p ≤ x`; the zeros enter only to
*characterise* the output (define the comparison window, name the GUE target) —
never as input.

## The readout: the spacing ratio r̃, not Σ²/Δ₃

`Σ²(L)` and `Δ₃(L)` need the spectrum unfolded by its own smooth count. At the
few-hundred-level scale this repo works at, that fitted unfolding is the weak
link — #9 found the cross-`x` `Δ₃(L)` ordering clean but `Σ²(L)` washed out by
the polynomial-unfold suppression (the #20 caveat). The Atas (2013) spacing ratio

    r̃_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}),   s_n = E_{n+1} − E_n

is taken on the **raw** levels, so the local density cancels and **no unfolding is
needed** (`spacing.spacing_ratios`, the #35 readout). References: ⟨r̃⟩ = 0.6027
(GUE 3×3 surmise), 0.5359 (GOE), 0.3863 (Poisson), → 1 for a rigid picket fence.
The first ~2000 real zeros give raw ⟨r̃⟩ ≈ 0.617 (#35) — decisively GUE.

## The forward result (N=160, dps=210)

⟨r̃⟩ over the full computed spectrum and over the low `40` (zero-tracking) window:

| x = λ² | 6 | 9 | 12 | 14 | 18 | 24 |
|---|---|---|---|---|---|---|
| full | 0.817 | 0.793 | 0.741 | 0.713 | 0.681 | 0.625 |
| low 40 | 0.705 | 0.655 | 0.624 | 0.624 | 0.624 | 0.624 |

Two monotone movements **toward GUE as the prime cutoff x grows**:

1. **The zero-tracking window converges and saturates.** The lowest ~40 levels —
   the ones that reproduce the true ordinates to many digits (§6 table) — relax to
   ⟨r̃⟩ ≈ 0.624 by `x ≈ 12` and then sit there: for `x ∈ {12,14,18,24}` the low-40
   levels *are* the first 40 zeros to high precision, so their spacing ratios are
   the zeros' (≈ the GUE/real-zero value). More primes don't change an
   already-converged window.
2. **The GUE-tracking range extends.** At fixed `N`, growing `x` raises the index
   `k*(x)` up to which the operator tracks zeros, so progressively *more* of the
   spectrum is GUE-like. By `x = 24` the whole `N = 160` spectrum is GUE-like
   (full ⟨r̃⟩ = 0.625 ≈ low). This is the #9 forward prediction stated cleanly:
   more primes ⇒ the operator's local statistics approach the zeros' GUE statistics
   out to higher energy. **`k*(x)` is now measured — see the next section (#53).**

## The tracking-range law `k*(x)` — measured (#53)

Movement 2 is now quantitative. Define the **zero-tracking range** `k*(x)` as the
length of the leading block of eigenvalues that reproduce the ordinates to a fixed
relative tolerance — `|ν_k − ζ_k| / ζ_k < 10⁻³` for every `k ≤ k*` — and the
**tracking height** `t*(x) = ζ_{k*}`, the energy up to which the operator tracks
(`ccm_convergence.tracking_length`; forward — the zeros only *score* the
prime-built spectrum). Swept at `N = 160` (one cached spectrum per `x` gives all
three readouts, so the axes align):

| x = λ² | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|
| `k*(x)` | 14 | 26 | 38 | 48 | 62 | 75 | 90 | 106 | 117 |
| `t* = ζ_{k*}` | 60.8 | 92.5 | 118.8 | 139.7 | 167.2 | 192.0 | 219.1 | 247.1 | 265.6 |
| `t*/x` | 10.1 | 11.6 | 11.9 | 11.6 | 11.9 | 12.0 | 12.2 | 12.4 | 12.1 |
| ⟨r̃⟩ full | 0.817 | 0.798 | 0.758 | 0.741 | 0.713 | 0.692 | 0.681 | 0.656 | 0.646 |
| ⟨r̃⟩ low 40 | 0.705 | 0.672 | 0.628 | 0.624 | 0.624 | 0.624 | 0.624 | 0.624 | 0.624 |
| Δ₃→GUE dist | 0.045 | 0.046 | 0.044 | 0.042 | 0.039 | 0.036 | 0.035 | 0.032 | 0.031 |

**The law: the tracking *height* is linear in the prime cutoff,** `t*(x) ≈ 12 x`.
The ratio `t*/x` sits in a tight band (mean ≈ 11.75, between `2π ≈ 6.28` and
`2πe ≈ 17.08`; the mild upward drift is a sub-leading correction and integer-`k*`
granularity). Equivalently `k*(x)` grows ~linearly, ≈ 6–7 newly-tracked zeros per
unit `x`. The origin is a **pole/zero density balance**: the secular poles
`d_n = 2πn/L` are uniform with spacing `2π/L = 2π/ln x`, while the zero spacing
`2π/ln(t/2π)` shrinks with height — the operator can resolve a zero only while the
poles stay denser, i.e. while `ln(t/2π) ≲ ln x`, a height `t ∝ x`. (The measured
`12` exceeds the bare crossover `2π` because "tracked to `10⁻³`" sits a little above
the exact density crossover; the *linearity* is the law, the constant is empirical.)

**The r̃ bridge — why this is the mechanism behind movements 1 & 2.** Read the
table down. As `x` grows, `k*(x)` climbs while the low-40 window has long since
saturated at ⟨r̃⟩ = 0.624 (movement 1 — by `x = 12` the low 40 *are* the first 40
zeros). The full-spectrum ⟨r̃⟩ then descends monotonically toward that same value
(0.817 → 0.646) **because** a growing fraction `k*(x)/N` of the levels are the
GUE-distributed zeros and not the pole-locked tail. At the looser "statistically
GUE" tolerance `10⁻¹`, `k*(x)` reaches ≈ `N` already by `x ≈ 16–18` — exactly where
full ⟨r̃⟩ closes on low ⟨r̃⟩. So the #18 r̃ descent and the `k*(x)` law are one
phenomenon. The lever is `x`, not `N`: enlarging `N` at fixed `x` leaves `k*(x)`
put and only lengthens the pole-locked tail (the push-`N` control below).

**Δ₃ ordering, quantified (#53 item 1).** The cross-`x` Δ₃(L) ordering the #9 read
saw by eye is now a number — the mean `|Δ₃_emp − Δ₃_GUE|` over the `L`-grid
(`spacing.delta3_gue_distance`) falls **33%** across the sweep (0.046 → 0.031),
monotone for `x ≥ 8` (the `x = 6` cutoff — only the primes {2,3,4,5} — is too coarse
to place). `Σ²` stays washed out under the polynomial unfold (the #20 caveat), so
the Δ₃-distance and the unfolding-free r̃ remain the trustworthy readouts; the
*absolute* Δ₃ level is still finite-`N`-suppressed.

The figure is `data/ccm_tracking_range.png` (`k*(x)` with the linear-law overlay;
the `t*/x` plateau).

## Two cautions baked into the design

- **Absolute rigidity is picket-*ward* at small x** (⟨r̃⟩ ≫ 0.603, var(s) ≪ 0.178),
  consistent with the deformed-`xp` near-picket fence (#24) and the zeros
  saturating *below* GUE at long range (#15). The trustworthy statements are the
  **cross-`x` trend** and the **zero-tracking-window convergence**, not the
  absolute level at any single small `x`.
- **Pushing `N` is counterproductive — read the low window, push the primes.** At
  fixed `x = 14`, ⟨r̃⟩ over the full spectrum *climbs away* from GUE with `N`
  (0.621 → 0.690 → 0.722 → 0.749 for `N = 60,120,180,240`). Above `k*(x)` the zero
  density outruns the pole spacing `2π/L`, so the secular roots become pole-locked
  (picket-like) and enlarging `N` just extends that non-GUE tail. The lever for the
  science is the prime cutoff `x` (with `N` only large enough to resolve `k*(x)`);
  the #18 eigensolve speedup matters because it makes the *x-sweep* cheap, not
  because it lets `N` grow.

## Reproduce

    uv run python scripts/run_ccm_gpu.py --mode universality \
        --x 6 9 12 14 18 24 --N 160 --pushn --pushn-x 14 --pushn-N 60 120 180 240

Writes `data/ccm_rtilde_vs_cutoff.png` (⟨r̃⟩ vs `x`, plus the push-`N` control) and
the existing `Σ²`/`Δ₃` rigidity + spacing figures. CPU/mpmath for the spectra; the
statistics are fp64 (GPU path falls back to numpy when CuPy is absent).

The `k*(x)` tracking-range law, the r̃ bridge, and the Δ₃-to-GUE ordering (#53):

    uv run python scripts/run_ccm_convergence.py --mode tracking \
        --tracking-x 6 8 10 12 14 16 18 20 22 --tracking-N 160

Writes `data/ccm_tracking_range.png` and prints the aligned table above. Spectra are
cached full-precision under `data/`, so the first run is the cost (~4 min here) and
reruns/replots are instant. Keep the largest `x` interior (`k* < N`); past that
`t*` is pinned at `ζ_N` and understates the height (shown hollow in the figure).
