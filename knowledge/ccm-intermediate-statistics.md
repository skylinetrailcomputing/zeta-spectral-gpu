# ccm-intermediate-statistics.md — the Šeba / rank-one read of the CCM tail (#87)

The flagship operator is a **rank-one perturbation of a scaling operator**
([`ccm-operator.md`](ccm-operator.md) §6), and quantum chaos has a named,
quantitative theory for exactly that operator class: point scatterers (Šeba
billiards) and the **intermediate spectral statistics** their rank-one secular
equation generates (Bogomolny–Gerland–Schmit). This note pins that theory onto
the operator's own secular equation and reports what the measurement said. It
upgrades the phenomenological "pole-locked / picket-like tail" language of
[`ccm-universality.md`](ccm-universality.md) into a parameter-free local theory —
and records which of the two boundaries in that note the theory can see.

Forward throughout: every input is the operator's own (the secular couplings and
poles, i.e. the minimal Weil eigenvector and the scaling spectrum); no ζ zero is
consumed anywhere. The zeros appear only as scorekeepers — the #53 tracking
height `t*` that the coupling-side boundary is compared against.

## The operator as a Šeba-class system

The spectrum of `D_log^{(λ,N)}` is the root set of the secular function
(`ccm.operator_eigenvalues`)

    F(z) = Σ_{n=-N}^{N} ξ_n / (d_n − z),    d_n = 2πn/L,

a **picket of poles** with uniform spacing `Δ = 2π/L` carrying signed,
prime-built couplings `ξ_n` (the components of the minimal even Weil
eigenvector). That is the Šeba / point-scatterer shape — with two deviations from
the textbook setup, both load-bearing:

- **the unperturbed levels are a picket, not Poisson.** BGS's celebrated
  semi-Poisson law `P(s) = 4s e^{−2s}` is the *daughter of a Poisson pole set*;
  with uniform poles the statistics are instead "picket with jitter", and
  semi-Poisson enters only as a reference marker. Its unfolding-free spacing
  ratio is **⟨r̃⟩ = ½ exactly**: consecutive semi-Poisson spacings are
  independent Γ(2) variables (each the sum of two fresh Poisson gaps), the ratio
  density is `6r/(1+r)⁴`, and `2∫₀¹ 6r²/(1+r)⁴ dr = ½` (verified against the
  simulated construction in the tests). Reference ladder: Poisson 0.386 —
  **semi-Poisson 0.500** — GUE 0.6027 — picket 1.
- **the couplings are signed** (eigenvector components, not `|c|²`), so the roots
  need *not* interlace the poles: a pole gap can hold 0, 1 or 2 roots. This is
  the theory-side explanation of what the root-finder in `ccm.py` observed
  empirically ("the roots do not interlace the poles … 0, 1 or occasionally 2").

The theory's content is **locality** — a root's position is set by the couplings
of the few poles around it, measured against `Δ`. Two levels, both parameter-free
(`ccm_intermediate.py`):

1. **First-order pinning**: a root attaches to pole `d_n` at offset
   `δ_n = ξ_n / R_n`, `R_n = Σ_{m≠n} ξ_m/(d_m − d_n)` (the smooth background of
   `F` at the pole), valid while `w_n = |δ_n|/Δ ≪ 1`.
2. **The local two-pole model**: in each gap keep the two flanking pole terms
   exactly and freeze the rest of `F` at its mid-gap value — a quadratic per
   gap, predicting both root **positions** and **gap occupancy** (0/1/2) with no
   root-finding on the full `F`.

## What the measurement said (N = 160, x = 6…22, #53 caches)

    uv run python scripts/run_ccm_convergence.py --mode intermediate

| x | occ. agree | med dz/Δ | ⟨r̃⟩ tail meas | ⟨r̃⟩ tail pred | w median | t_dens/x | t*/x |
|---|---|---|---|---|---|---|---|
| 6  | 100% | 0.022 | 0.836 | 0.818 | 0.59 | 6.7 | 10.1 |
| 10 | 100% | 0.020 | 0.800 | 0.785 | 0.40 | 6.7 | 11.9 |
| 14 | 99%  | 0.023 | 0.773 | 0.750 | 0.40 | 6.4 | 11.9 |
| 18 | 98%  | 0.023 | 0.766 | 0.743 | 0.30 | 9.5 | 12.2 |
| 22 | 99%  | 0.027 | 0.760 | 0.740 | 0.36 | 6.3 | 12.1 |

(`occ. agree` = fraction of the 160 pole gaps where the local model's predicted
occupancy equals the measured spectrum's; `med dz/Δ` = median distance from each
predicted root to the nearest measured one; tail = above `t*`; `t_dens` = the
deficit-plateau midpoint, below.)

**1. Locality is the tail's statistics — quantitatively.** The two-pole local
model reproduces the measured gap occupancy in 98–100% of gaps (at `x = 6` and
`x = 10`, all 160 — including every 0- and 2-root defect) and the root positions
to a median ~0.02 Δ, and its windowed ⟨r̃⟩ curve lies within ~0.02–0.03 of the
measured one everywhere. "Pole-locked / picket-like" is now a theory whose
failure would have been informative; it didn't fail.

**2. The tail is *intermediate*, not the trivially-pinned picket.** The natural
expectation — couplings decay up the spectrum, so the tail approaches the
`w → 0` picket — is wrong. The background `R_n` decays at the *same rate* as the
couplings (the mode-sum `Σξ_n` nearly cancels, so `R_n` is dominated by the
neighboring couplings, not the big low modes), and the dimensionless coupling
stays **scale-free at `w ~ 0.3–0.6`** down the whole tail at every cutoff. The
tail is a picket with O(1)-correlated jitter: ⟨r̃⟩ ≈ 0.76–0.84, far picket-ward
of GUE but well short of 1, drifting only slowly with `x`. This is where the
operator sits "between picket and semi-Poisson": at a roughly cutoff-independent
intermediate point fixed by the coupling-to-background ratio — the answer to
the question the issue posed.

**3. Of the two boundaries, the local read sees the density crossover — not
`t*`.** Below `t ≈ 2πx` the root density (which follows the zero density where
it can) is smaller than the uniform pole density, so empty gaps accumulate; the
cumulative root-vs-pole deficit grows, peaks, and is then paid back by 2-root
gaps. That **deficit plateau** (`ccm_intermediate.deficit_plateau`), computed
from the *predicted* occupancy alone, straddles the density-crossover line:
plateau midpoints `t_dens/x ≈ 6.3–9.5`, mean ≈ 7.1, bracketing `2π ≈ 6.28` —
a coupling-side, zero-free recovery of the `2πx` law. The **tracking height
`t* ≈ 12x`** (#53), by contrast, is *invisible* to every local readout tried:
the `w_n` profile shows no break at `t*`, occupancy defects persist above it,
and the local model is equally accurate on both sides. The honest conclusion of
the issue's "second handle on `k*(x)`" hope: zero-tracking is not encoded in
local coupling *magnitudes* — it lives in the long-range correlated *pattern*
of the couplings (the same information that makes the low spectrum reproduce
the zeros), which single-gap statistics cannot see. Both halves are findings:
the `2π` line falls out of the couplings; the `12x` line genuinely needs the
zeros to score.

**4. A precision finding, en route.** The background sums `R_n` / `B` cancel
from O(0.1)-size low-mode terms down to the scale of the local couplings
(`1e-25` and below in the deep tail). In float64 they are pure roundoff —
the first-cut fp64 implementation produced exactly the kind of plausible-looking
garbage (`w` alternating between `1e-5` and `1e+2`) that #65 documented for the
spectrum itself. The analysis layer therefore runs its sums in mpmath at the
`ξ`-cache's working precision and only casts the O(1)-conditioned ratios to
float64. Same wall, new corner: **everything downstream of `ξ` that subtracts
cancels, not just the eigensolve.**

The figure is `data/ccm_intermediate_stats.png` (coupling profile, windowed ⟨r̃⟩
measured vs theory, pooled tail spacing histogram, and the two boundary laws).
Spectra and eigenvectors are cached under `data/` keyed by `(N, x, dps)` — the
`--mode intermediate` run reuses the #53 spectrum caches and adds `ξ` caches
(`ccm_xi_mpf_*.json`), so reruns are seconds.

## Relation to the other CCM notes

- [`ccm-universality.md`](ccm-universality.md) — measured the two boundaries
  this note explains/locates: the `k*(x)`/`t* ≈ 12x` tracking law (zero-scored)
  and the pole/zero density-balance argument (whose `2πx` crossover the
  occupancy deficit now reads out forward). The push-`N` caution ("enlarging `N`
  only lengthens the pole-locked tail") gets its mechanism: the added tail is
  the same scale-free intermediate point process, so it dilutes the GUE fraction
  without ever becoming literally rigid.
- [`ccm-convergence-law.md`](ccm-convergence-law.md) / #82 — the edge
  eigenvalues being "pinned to the bulk pole `d_n`" (the fp64-robustness
  signature) is the `w < 1` side of the same local structure; the low-band
  corruption lives exactly where locality is weakest.
- [`quantum-chaos-map.md`](quantum-chaos-map.md) — this fills the
  intermediate-statistics row of the dictionary: the one QC tool aimed at the
  repo's *own operator* rather than at the zeros.

## Sources

- P. Šeba, *Wave chaos in singular quantum billiard*, Phys. Rev. Lett. **64**
  (1990) 1855 — the rank-one point-scatterer class.
- E. Bogomolny, U. Gerland & C. Schmit, *Models of intermediate spectral
  statistics*, Phys. Rev. E **59** (1999) R1315 — semi-Poisson and the
  rank-one secular-equation statistics.
- E. Bogomolny, U. Gerland & C. Schmit, *Singular statistics*, Phys. Rev. E
  **63** (2001) 036206 — the Šeba-billiard statistics proper (strong-coupling
  point scatterer).
- G. Berkolaiko, E. Bogomolny & J.P. Keating, *Star graphs and Šeba billiards*,
  J. Phys. A: Math. Gen. **34** (2001) 335 — the star-graph realization of the
  same intermediate class.
- Y.Y. Atas, E. Bogomolny, O. Giraud & G. Roux, Phys. Rev. Lett. **110** (2013)
  084101 — the `r̃` machinery (`spacing.py`); same first author as the
  intermediate-statistics line.
- arXiv:2511.22755 (Connes–Consani–Moscovici) — the operator itself; see
  [`ccm-operator.md`](ccm-operator.md) and [`bibliography.md`](bibliography.md).
