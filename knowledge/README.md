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
- [`ccm-operator.md`](ccm-operator.md) — the verified, code-ready spec for the
  flagship Connes–Consani–Moscovici finite-cutoff operator, pinned
  equation-by-equation against arXiv:2511.22755. Read this before implementing
  the flagship matrix fill or λ-sweep.

The deeper mathematical background (harmonic functions → conical/Legendre
functions → hyperbolic/Selberg spectrum → GUE statistics, and the full
Hilbert–Pólya landscape) lives in the predecessor repo
[`wedgetrigfunctions202601`](https://github.com/bradleypmartin/wedgetrigfunctions202601)
under its own `knowledge/`. That arc is treated as settled background here.
