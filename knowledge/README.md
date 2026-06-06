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
- [`ccm-universality.md`](ccm-universality.md) — the prime-cutoff rigidity trend
  (#18): the unfolding-free spacing-ratio `r̃` read of the operator spectrum. As
  the cutoff `x` grows the local statistics relax toward GUE and the
  zero-tracking window converges to the real zeros' `⟨r̃⟩`; why pushing `N` is the
  wrong lever (the pole-locked tail).
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
