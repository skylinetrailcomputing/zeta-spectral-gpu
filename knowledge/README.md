# knowledge/

Conceptual, cross-session notes — *why* the math looks the way it does and what
the project is allowed to do. Operational "how to run things" lives in the
top-level `CLAUDE.md`; per-session narrative lives in git history.

- [`project-framing.md`](project-framing.md) — the forward-vs-inverse rule and
  the two-phase roadmap (warm-up statistics → Connes–Moscovici flagship). Read
  this before adding any experiment.
- [`debruijn-newman-flow.md`](debruijn-newman-flow.md) — why the De Bruijn–Newman
  heat flow is a *forward* rigidity experiment (the warm-up spike #20), and the
  precision lesson it shares with the flagship.
- [`deformed-xp.md`](deformed-xp.md) — Sierra's geometrically deformed `xp`
  operator (#23/#24/#31): it reproduces the zeros' **mean density** but sits at a
  rigid picket fence, **not** GUE — the clean forward demonstration that matching
  the average is necessary but nowhere near sufficient. First entry in the
  xp-Hamiltonian track.
- [`dirac-mirror.md`](dirac-mirror.md) — Sierra's prime-driven Möbius-mirror
  forward **locator** (#25): `|M'_z(n)|` peaks at the ζ (and Dirichlet-`L`) zeros.
  Why its peak *spacings* are circular (no single-operator spectrum), the four
  landed phases incl. the RH-by-contradiction demo, and the open forward
  fluctuation spike #44.
- [`ccm-operator.md`](ccm-operator.md) — the verified, code-ready spec for the
  flagship Connes–Consani–Moscovici finite-cutoff operator, pinned
  equation-by-equation against arXiv:2511.22755. Read this before implementing
  the flagship matrix fill or λ-sweep.
- [`connes-cvs-oracle.md`](connes-cvs-oracle.md) — the `connes-cvs` baseline
  oracle (#16): the documented `c=13` `λ_min` cell, frozen as a checked-in
  fixture for the CPU reference (#8) and GPU assembly (#9) to diff against, plus
  the `λ_min` vs zero-error distinction and the ~1.2–1.3 normalization factor.
- [`ccm-reproduction-notes.md`](ccm-reproduction-notes.md) — a self-contained,
  public-facing field guide (#56) for anyone reproducing the CCM / Connes–van
  Suijlekom zeta operator: what we reproduced (the `connes-cvs` `c=13` cell + the
  §6 table from our own assembly) and the five gotchas a re-implementer hits — the
  Weil combination **sign** (`W_{0,2} − W_ℝ − Σ_p W_p`, not all-`+`), `λ_min` vs
  zero-error + the ~1.2–1.3 normalization factor, the end-to-end precision wall
  (fp64 *corrupts* the spectrum), the factor-once / parity-reduced eigensolve
  recipe, and the small construction traps. Consolidates the issue-numbered notes
  below for an outside (Connes/CvS-camp) reader; outreach is gated on maintainer
  sign-off.
- [`ccm-universality.md`](ccm-universality.md) — the prime-cutoff rigidity trend
  (#18): the unfolding-free spacing-ratio `r̃` read of the operator spectrum. As
  the cutoff `x` grows the local statistics relax toward GUE and the
  zero-tracking window converges to the real zeros' `⟨r̃⟩`; why pushing `N` is the
  wrong lever (the pole-locked tail).
- [`ccm-intermediate-statistics.md`](ccm-intermediate-statistics.md) — the Šeba /
  rank-one intermediate-statistics read of the CCM pole-locked tail (#87): the
  secular equation as a point-scatterer system (signed couplings ⇒ no
  interlacing), the parameter-free local two-pole model that reproduces the
  measured gap occupancy (98–100%) and tail statistics, the scale-free
  intermediate coupling `w ~ 0.3–0.6` (the tail never reaches the picket), and
  the two boundaries: the `2πx` density crossover falls out of the couplings,
  the `t* ≈ 12x` tracking law does not (it lives in coupling correlations).
- [`ccm-fill-precision.md`](ccm-fill-precision.md) — the precision anatomy of the
  flagship Weil-fill and why a **double-double fill is a no** (#54): ~99% of the
  fp64 matrix error is the special-function coefficients, not the fill arithmetic
  (Sterbenz makes the near-band divided-difference subtraction exact). A dd fill
  over fp64 coefficients gains ~0 digits, dd's ~32 digits is below the ~80 needed
  at `c = 13`, and the fp64 `eigh` consumer can't even use the 15 digits the fill
  already has. The real lever is the coefficient sweep at qd/bignum *paired with*
  an extended-precision eigensolve, not the assembly fill.
- [`riemann-siegel.md`](riemann-siegel.md) — the GPU Riemann–Siegel ζ-evaluator
  (#55): a forward-neutral *tool* computing `ζ(1/2 + i t)` / the Hardy `Z` from the
  Riemann–Siegel expansion in fp64. The `O(√t)` main sum is the embarrassingly-
  parallel CUDA target (~131× at `t = 1e8`); `Ψ`'s derivatives go through a Cauchy
  contour to survive its removable singularities, and fp64 has a phase-argument
  height ceiling (~7 digits by `t = 1e8`). Unlocks zero verification (#51/#60),
  De Bruijn–Newman at height (#20), and value-distribution work.
- [`frontier-survey-2026.md`](frontier-survey-2026.md) — the #50 literature
  coverage check (2023–2026 forward spectral / Hilbert–Pólya numerics): the
  Connes-lineage frontier (incl. the 2026 Groskin `connes-cvs` implementation
  arXiv:2605.20224 and Śliwiński's inverse-log convergence law arXiv:2601.12133),
  verdicts on the four self-identified gaps (Katz–Sarnak #51, Li #52, BBM,
  Keating–Snaith), and a ranked list of forward experiment candidates.
- [`katz-sarnak-families.md`](katz-sarnak-families.md) — Katz–Sarnak family
  statistics (#51): the quadratic Dirichlet family `L(s, χ_d)` (Kronecker
  characters over fundamental discriminants) is **symplectic** — its conductor-
  rescaled low-lying zeros are suppressed at the central point, tracking
  `1 − sin(2πx)/(2πx)` and away from unitary/orthogonal. The forward family
  companion to the single-sequence GUE warm-up; mpmath ground truth + the GPU
  locator as the embarrassingly-parallel-over-the-family producer.
- [`li-criterion.md`](li-criterion.md) — Li's criterion (#52): RH ⟺ `λ_n ≥ 0`,
  computed **forward** from the `log ξ` Taylor coefficients (Stieltjes constants +
  polygamma; Bombieri–Lagarias), never the `Σ_ρ` sum over zeros. A cheap **scalar
  shadow** of the flagship's Weil positivity — the swept `λ_n` come out positive,
  tracking the RH growth law `(n/2)(log n + γ − 1 − log 2π)`. mpmath-bound
  (cancellation + Stieltjes precision); the GPU angle is the parallel-over-family
  GRH sweep `λ_n(χ)`, not a deeper single-`ζ` pass.
- [`quantum-chaos-map.md`](quantum-chaos-map.md) — the quantum-chaos camp's
  toolkit keyed to this repo's tracks (#88): the Gutzwiller ↔ explicit-formula
  dictionary made explicit (orbits = primes, periods = `log p`, amplitudes =
  `Λ(n)/√n`), what the camp adds beyond universal RMT (the lower-order
  arithmetic ladder #84; intermediate statistics for the CCM tail #87), and
  what stays out of charter (moments, BBM, quantum graphs).
- [`arithmetic-correlations.md`](arithmetic-correlations.md) — arithmetic
  beyond universality (#84): the Conrey–Snaith / Bogomolny–Keating lower-order
  pair-correlation terms (two published forms pinned and proven equal in the
  tests) and the explicit-formula prime peaks of the zeros' Fourier transform
  `S(u)` at `u = log p^m`. The zeros' measured departure from the bare sine
  kernel matches the prime-side prediction at shot-noise level already at
  height ~5·10⁴, and below the Heisenberg frequency `|S(u)|²` is
  spike-dominated (the GUE ramp is only its smoothing) — the primes literally
  visible in the zeros.
- [`davenport-heilbronn-control.md`](davenport-heilbronn-control.md) — the
  negative control (#85): the forward machinery run on the Davenport–Heilbronn
  function (exact functional equation, **no Euler product**, provable RH
  violations). The off-line zeros are censused as output (first at
  `0.8085 + 85.6993i`, matching the published tables), the #43 growth law
  fires on a *genuine* off-line zero (slope ≈ `σ_c − ½`), and the headline:
  **local spacing statistics are blind** (f keeps GUE-level repulsion at
  modest height — the mirror of #87's operator-side lesson) while the
  smooth-count deficit, the growth exponent, and the locator's off-line
  mounds are where the pipeline genuinely distinguishes f from ζ.
- [`bibliography.md`](bibliography.md) — the primary source papers behind the
  experiments, with stable arXiv links and which note/issue uses each. The
  tracked index of the (gitignored) local PDF cache, so contributors can see the
  sources and the cache is rebuildable. Per-note `Sources` sections still hold the
  secondary literature.
- [`predecessor/`](predecessor/) — distilled, public-safe archive of the CPU
  predecessor `wedgetrigfunctions202601` (private): the harmonic-functions →
  conical → hyperbolic/Selberg → GUE arc, and the arithmetic-chaos headline
  (Riemann zeros = GUE, modular Maass = Poisson, via Hecke symmetries) that
  motivates this repo's warm-up phase and its forward-not-inverse rule. Settled
  background — the prerequisite reading.

The deeper mathematical background (harmonic functions → conical/Legendre
functions → hyperbolic/Selberg spectrum → GUE statistics, and the full
Hilbert–Pólya landscape) comes from that predecessor. Its repo is private, so
the public-safe distillation in [`predecessor/`](predecessor/) is the in-repo
stand-in; that arc is treated as settled background here.
