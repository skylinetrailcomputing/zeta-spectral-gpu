# ccm-universality.md — the prime-cutoff rigidity trend (#18, building on #9/#35)

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
   out to higher energy.

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
