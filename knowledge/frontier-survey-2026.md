# Frontier survey: forward spectral / Hilbert–Pólya RH numerics (2023–2026)

A coverage check of the recent literature, run because this repo's lineage
tracking stopped at a Jan-2026 knowledge cutoff — exactly where the live
Connes-lineage frontier sits. The question (issue #50): what *forward and
numerically computable* spectral / Hilbert–Pólya / quantum-chaos work has
appeared, and which candidate experiments survive the
[forward-not-inverse rule](project-framing.md)?

> **Method & caveat.** Produced by a fan-out web pass (multi-angle search →
> primary-source extraction → 3-vote adversarial verification → synthesis;
> 24 primary sources, 24/25 sampled claims confirmed). **Every 2026 item below
> is an un-refereed preprint.** And the load-bearing honest fact for the whole
> field: across the Connes lineage, *criticality* — the truncated form's
> ground-state zeros provably lie on the critical line — is **proven**, but
> **convergence of the operator spectra to the actual ζ zeros remains open.**

## 1. The Connes lineage is the live frontier

The flagship's source papers (CCM *Zeta Spectral Triples* arXiv:2511.22755;
Connes–van Suijlekom arXiv:2511.23257; Connes 2026 arXiv:2602.04022; framework
arXiv:2106.01715; prolate arXiv:2310.18423) remain the state of the art for a
*forward* operator whose spectrum is conjectured to converge to the zeros. Two
2026 preprints extend the numerical/analytic picture and sit **past the training
cutoff**:

- **Groskin, arXiv:2605.20224** (13 May 2026), *High-Precision Approximation of
  Riemann Zeros via the Truncated Weil Form.* **Forward.** The first public
  implementation of the Connes–van Suijlekom Galerkin matrix — and the paper
  behind the `connes-cvs` package this repo already uses as its
  [oracle](connes-cvs-oracle.md). Sixteen cutoffs (`c = 13–67`, plus `100`),
  matrix dimension `N = 100–250`, precision `dps = 500–1000`; the first-zero
  absolute error shrinks monotonically `~2e-55 → ~1.5e-168` across cutoffs and
  recovers `γ₁…γ₁₀` to 307–329 matching digits. Built from the primes `p ≤ c`;
  "we make no claim of proof"; convergence to the true zeros explicitly left
  open. Its precision regime matches the ~200-digit / ~1e-55 figure already noted
  in [`project-framing.md`](project-framing.md).
- **Śliwiński, arXiv:2601.12133** (17 Jan 2026), *Spectral Analysis of the
  D_log^(λ,N) Operators.* **Forward.** Shows that under several standard error
  notions the dissonance between the CCM spectra and the ζ zeros is
  **inverse-logarithmic**, "elegantly fitting the distribution of primes." This
  is the most *actionable* find: it supplies the convergence law a
  sequence-acceleration scheme would extrapolate against (candidate **F2** below).

## 2. The four self-identified gaps, checked against the literature

| Strand | Verdict |
|---|---|
| **Katz–Sarnak family statistics** (low-lying-zero symmetry type for families of *L*-functions; issue #51) | **Confirmed the strongest forward thread — with a numerical niche wide open.** Recent work (Dillon–Miller et al. arXiv:2509.05810 on GL(2) one-level-density moments; arXiv:2408.09050; Maass arXiv:2505.18712) is **purely analytic** — it proves n-level-density moment formulas, it does not compute them at scale. Forward by the same logic as the GUE warm-up: statistics taken from the arithmetic/prime side and compared to a compact-group prediction; the zeros are never fit. Typically GRH-conditional. |
| **Li's criterion** (positivity of λ_n from `log ξ`; issue #52) | Forward-legitimate and tractable, but **no new 2026 development surfaced**. Mature existing tooling (arXiv:1703.02844; fredrikj.net). Lower novelty — a scalar shadow of the flagship's Weil positivity. |
| **Bender–Brody–Müller (2017)** PT-symmetric Hamiltonian | **Exclusion confirmed, on two grounds.** Sierra's review (arXiv:1601.01797 §IX) records the BBM operator as **non-self-adjoint**; and it has no independent forward derivation — the eigenvalues-are-zeros identification leans on ζ itself. No 2023–2026 rehabilitation found. |
| **Keating–Snaith / CFKRS moments** | **Correctly out of charter.** Recent moments work (arXiv:2509.07788, arXiv:2405.06474, arXiv:2301.10634) characterizes the *value distribution* of ζ / *L*-functions, not an operator spectrum. Nothing makes it a forward *spectral* target. |

## 3. One new adjacent operator family

**Hateley, arXiv:2511.18309** (23 Nov 2025), *A Chiral Adelic Dirac Operator and
the Spectral Realization of the Riemann Zeros.* **Gate cleared — forward; closed
as won't-build (issue #66).** A chiral adelic Dirac operator on the idèle class
space with a prime-indexed Hecke mass deformation; the zeros appear not as a raw
spectrum but as **spectral-shift-function discontinuities**, and a separated
adelic trace formula yields a prime-orbit expansion echoing the explicit formula.
Finite-prime truncations are computable.

The full-text watch-item — confirm no ζ/zero data tunes the mass term, per the
#44 circularity precedent — is **discharged**: it does not. The functional
equation enters structurally as the chiral involution `J_glob` (reflection ∘
idelic inversion); the offered coefficient families (`η_p=p^{−(1+ε)}λ²`, and the
formal Euler-factor `−[log(1−p^{−s})]′|_{1/2}`) are prime-built; the only contact
with the zeros is a global 2-parameter affine rescale at comparison time
(Berry–Keating-style), not the #44 trap of feeding in `arg ζ`.

It is nonetheless a **no-build**: (i) the forward coefficient families don't
reproduce the zeros — §5.2's numerics use *random synthetic* Hecke modes and
report MAE ≈ 7–17 vs. the first 20 zeros with **no convergence** in primes or
Hecke-mode count, author-disclaimed as non-evidentiary; (ii) the natural ζ seed
(`λ_p≡1`) collapses the mass to a constant ⇒ rigidly-shifted Floquet bands with
no per-zero structure; (iii) the actual zero-matching is **Conjecture 1**, an
*existence* claim about `η_p` chosen to match the zeros — searching for those is
inverse (Wu–Sprung with Hecke dressing); (iv) the truncated model is a trivial
2×2 fiber `±|E_n(κ)−E_*+m|` with no eigensolve/precision/GPU surface. It adds
nothing past the prime-built Möbius-mirror locator (#25) or the CCM flagship.
Adjacent to the [Dirac-mirror track](dirac-mirror.md).

## 4. Ranked forward-experiment candidates

Ranked by (a) forward-legitimacy, (b) novelty vs. what the repo already has,
(c) GPU-parallelizability / precision tractability. The fp64 reality holds
throughout: low zeros can need precision beyond fp64, so heavy eigensolves stay
multiprecision on CPU; the GPU's leverage is parallel statistics and operator
*assembly*.

1. **F1 — Katz–Sarnak family statistics on GPU** (≈ #51). Evaluate a *family* of
   Dirichlet / GL(2) *L*-functions, extract low-lying zeros, compare the density
   near the central point to the symplectic / orthogonal / unitary prediction.
   **Forward.** *Novelty: high* — the theory papers prove formulas but do not
   compute at scale; this is the numeric companion to arXiv:2509.05810.
   *GPU: excellent* — embarrassingly parallel over family members, modest
   per-member precision.
2. **F2 — sequence-acceleration of the CCM spectrum** against Śliwiński's
   inverse-log law (the "bootstrap V3" idea on #50). Richardson / Shanks / Wynn-ε
   extrapolation of the repo's own cutoff-sequence of eigenvalues toward the
   `c → ∞` limit. **Forward** (extrapolates the repo's own forward sequence).
   *Novelty: moderate-high* — arXiv:2601.12133 characterizes the rate; nobody has
   built the accelerator or tested the law empirically. *GPU: high* (parallel
   multi-cutoff assembly; eigensolve stays CPU/multiprecision).
3. **F3 — coarse-to-fine Newton polish of CCM eigenvalues** (the "bootstrap V2"
   idea). Seed each cutoff's eigenvalues from the previous; polish a prime-defined
   characteristic root. **Forward.** *Novelty: medium* — Groskin already reaches
   `1e-168` by brute high-`dps`, so the win is **speed, not reach**. Pairs with F2.
4. **F4 — empirically test the inverse-log convergence law** on GPU-assembled CCM
   matrices (more cutoffs / larger `N`, fit the error law). **Forward.** Ties
   arXiv:2601.12133 and arXiv:2605.20224 together; same cluster as F2.
5. **F5 — explicit-formula self-consistency, "primes as truth"** (the "bootstrap
   V1" idea). **Forward** in principle; *thinnest foothold* — high risk of
   collapsing into "just evaluate ζ" (→ the Riemann–Siegel evaluator direction,
   #55). Capture, do not prioritize.
6. **F6 — Hateley adelic-Dirac toy implementation.** **Resolved: gate cleared
   (forward, not the #44 trap), closed as won't-build (#66).** The §3 watch-item
   is discharged — no ζ/zero data tunes the mass term. But the forward coefficient
   families don't reproduce the zeros (random synthetic Hecke data, MAE ≈ 7–17, no
   convergence, author-disclaimed); the matching is deferred to an inverse
   existence conjecture; and the truncated model is a trivial 2×2 fiber with no
   eigensolve/GPU surface. Adds nothing past #25/the flagship. (Full ruling:
   maintainer `_private/issue-66-hateley-ruling.md`.)

The litmus that tags F2–F5 as forward despite their iterative shape: the only
legal external anchor in the loop is the **primes** (or a prime-defined
function / certificate), never the true zeros; intermediate zero estimates may
serve as a disposable *seed* but never as a *fitting target*. Corrupt the seed
and a forward scheme re-converges to the same prime-defined fixed point — an
inverse one tracks the seed. (Full framing: issue #50 discussion.)

## 5. What was likely missed at the Jan-2026 cutoff

- **Groskin arXiv:2605.20224** (May 2026) — the `connes-cvs` oracle's paper, with
  its specific high-precision convergence figures.
- **Śliwiński arXiv:2601.12133** (Jan 2026) — the inverse-log convergence-rate
  result; the single most useful output of the pass for the flagship.
- **Hateley arXiv:2511.18309** (Nov 2025) — the chiral adelic Dirac family.

None overturns the forward-not-inverse charter or the flagship's standing; the
Connes lineage remains the live forward frontier, convergence-to-the-zeros still
open. The actionable updates are F1 (a genuinely open numerical niche) and the
F2/F4 acceleration cluster (now with a convergence law to aim at).

## References

Connes–Consani–Moscovici, arXiv:2511.22755 · Connes–van Suijlekom,
arXiv:2511.23257 · Connes, arXiv:2602.04022 · Connes–Consani, arXiv:2106.01715 ·
CCM, arXiv:2310.18423 · Groskin, arXiv:2605.20224 · Śliwiński, arXiv:2601.12133 ·
Hateley, arXiv:2511.18309 · Dillon–Miller et al., arXiv:2509.05810 ·
arXiv:2408.09050 · Maass, arXiv:2505.18712 · Sierra (review), arXiv:1601.01797 ·
Li-criterion numerics, arXiv:1703.02844 · moments, arXiv:2509.07788 /
arXiv:2405.06474 / arXiv:2301.10634.
