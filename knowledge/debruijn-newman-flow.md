# The De Bruijn–Newman flow as a *forward* rigidity experiment

This note explains why the De Bruijn–Newman (DBN) heat flow belongs in this repo
(it is forward, not inverse) and what the warm-up spike (#20) measures. Read
[`project-framing.md`](project-framing.md) first for the forward-vs-inverse rule.

## The object

For each real `t`, de Bruijn and Newman define an entire function as the cosine
transform of a heat-evolved, theta-derived kernel:

```
H_t(z) = ∫₀^∞ e^{t u²} Φ(u) cos(z u) du,
Φ(u)   = Σ_{n≥1} (2 π² n⁴ e^{9u} − 3 π n² e^{5u}) exp(−π n² e^{4u}).
```

At `t = 0` this is the Riemann ξ on the critical line, `H₀(z) = ⅛ ξ(½ + i z/2)`,
so the zeros of `H₀` sit at `z = 2γ` for each ordinate `γ` of a nontrivial zeta
zero. The **De Bruijn–Newman constant** `Λ` is defined by: `H_t` has *only real
zeros* iff `t ≥ Λ`. Known bounds: `Λ ≥ 0` (Rodgers–Tao 2018), `Λ ≤ 0.22`
(Polymath15 2019). Since RH ⟺ `Λ ≤ 0`, RH ⟺ `Λ = 0`.

Source conventions are pinned to Polymath15, *Effective approximation of heat
flow evolution of the Riemann ξ function* (arXiv:1904.12438); see the local PDF
cache.

## Why this is forward, not inverse

The discriminating test is *"does it consume the zeros as input?"* Here the
answer is no:

- We evaluate `H_t(z)` **directly from the structural kernel Φ** and root-find
  its real zeros. The zeros are an **output** we then characterise.
- We do **not** take the known zeta zeros as initial data and evolve them under
  the DBN "particle dynamics" (the backward heat-flow ODE on the zeros). That
  *backward* use is how one bounds `Λ` *from* the zeros — and it consumes the
  zeros, so it is the inverse trap this repo bans.

So the forward generator (`debruijn_newman.h_t_zeros`) and the backward
particle flow are mathematically related but sit on opposite sides of the rule.
We only ever do the forward one.

## What the flow does to rigidity (the physics)

`H_t` satisfies the backward heat equation `∂_t H = −∂_{zz} H`, whose zero
dynamics is *repulsive*: as `t` increases the real zeros spread toward a uniform
"picket fence." So for `t ≥ 0` the spectrum gets **monotonically more rigid** as
`t` grows — number variance `Σ²(L)` and the Dyson–Mehta `Δ₃(L)` *fall*, and the
nearest-neighbour spacing distribution narrows toward a delta at the mean.

This is **not** a GUE → GOE → Poisson symmetry-class crossover (the framing the
original stretch breadcrumb #7 got wrong). It is increasing rigidity *within* the
same real-zero family — the zeros stay real for all `t ≥ 0` (since `Λ ≤ 0` under
RH) and merely become more regular.

The readout is exactly the #15 statistics, applied to each `H_t` zero set after
**empirical unfolding** (each set unfolded by its *own* smooth count, not the
fixed `t = 0` Riemann–von Mangoldt `N̄`, so a trivial density change cannot
masquerade as a rigidity change).

## The spike (#20) and its precision lesson

The spike de-risks a full DBN investigation by asking only: *can we generate
`H_t` zeros at enough height to see the rigidity trend?* It is a **CPU/`mpmath`**
job, not a GPU one — `H_t` at height `z` is exponentially small
(`|H₀(2γ)| ~ exp(−π γ/4)`) relative to its `O(1)` integrand, so the oscillatory
integral cancels catastrophically and the bottleneck is **digits, not parallel
work** (consistent with the precision caveat in `CLAUDE.md`). The forward
generator runs in extended precision sized to the height
(`dps ≈ π γ_max / (4 ln 10) + guard`).

This is the same precision moral as the flagship and #15: at height the naive
forward integral is correct but costly, and pushing to the small-`t` regime
relevant to `Λ` (where the effect on low zeros is tiny) is precisely what would
need the Riemann–Siegel / Polymath effective expansion in a full 7b. See the
spike script `scripts/run_debruijn_newman.py` and issue #20 for the empirical
verdict.
