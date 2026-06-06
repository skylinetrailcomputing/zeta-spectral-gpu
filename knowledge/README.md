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

The deeper mathematical background (harmonic functions → conical/Legendre
functions → hyperbolic/Selberg spectrum → GUE statistics, and the full
Hilbert–Pólya landscape) lives in the predecessor repo
[`wedgetrigfunctions202601`](https://github.com/bradleypmartin/wedgetrigfunctions202601)
under its own `knowledge/`. That arc is treated as settled background here.
