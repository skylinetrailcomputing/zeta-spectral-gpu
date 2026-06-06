# Predecessor archive — `wedgetrigfunctions202601`

A self-contained, public-safe distillation of the conceptual notes and findings
from this project's CPU predecessor, the pure-Python research repo
**`wedgetrigfunctions202601`**. That repo is **private**, so its GitHub link
would 404 for anyone but the maintainer; this folder is the in-repo archive that
replaces it as the **prerequisite reading**. Its research arc is treated as
*settled background* here — this repo (`zeta-spectral-gpu`) reuses the
conclusions and scales the compute on GPU.

> **Provenance & attribution.** `wedgetrigfunctions202601` is the maintainer's
> own prior repo (no third-party code; its content is mathematical and
> public-safe). These files distill — not copy — its `knowledge/` docs and
> `CLAUDE.md`. The one external dataset it relied on, the LMFDB rigorous Maass
> spectrum, is *referenced* (cite by label + [arXiv:2502.01442](https://arxiv.org/abs/2502.01442);
> CC BY-SA 4.0), never redistributed here. The original Python scripts (Python
> 3.9 + `scipy`/`mpmath`) stay in the predecessor repo; they were CPU sandboxes,
> not forward experiments in this repo's sense, so they are summarized rather
> than ported.

## The arc in one sentence

Harmonic functions on a 2D wedge → 3D Euclidean cone → compact self-adjoint
hyperbolic **disk** / **annulus** → the modular-surface **Maass spectrum**
(LMFDB ingest + a from-scratch **Hejhal** solver) → spectral **Selberg zeta** →
**GUE nearest-neighbour spacing** — establishing that the critical line is forced
by self-adjointness, but that *only* the Riemann zeros (no decorating
symmetries) actually carry the GUE fluctuations, while the arithmetic modular
surface (Hecke operators) relaxes to Poisson.

## Contents

- [`theory-map.md`](theory-map.md) — **why** $\operatorname{Re}(s) = \tfrac12$
  recurs: harmonic functions → conical (Mehler–Fock) functions → the critical
  line via Mellin / scaling self-duality → the four pillars (Selberg proven,
  Hilbert–Pólya conjectural, Montgomery–Dyson empirical, Connes at the edge).
- [`computational-arc.md`](computational-arc.md) — **what** the predecessor
  computed at each geometry, the per-stage findings (Weyl laws, the
  modular-surface/Hejhal work), and the headline **GUE-vs-Poisson** result with
  its **arithmetic-chaos** explanation — the single most load-bearing learning
  for this repo's framing.

## Why this matters for `zeta-spectral-gpu`

The predecessor's headline — Riemann zeros are textbook GUE, the
$\mathrm{PSL}_2(\mathbb{Z})$ Maass spectrum is Poisson-leaning because Hecke
operators over-symmetrize it — is what motivates three things here:

1. **The warm-up phase.** GPU spacing / pair-correlation / rigidity statistics
   directly extend the predecessor's `gue_spacing.py`. See
   [`../project-framing.md`](../project-framing.md).
2. **The forward-not-inverse rule.** Matching the *mean density* is cheap and
   necessary; carrying the *GUE fluctuations* is the hard, discriminating part —
   the same negative recurs with the deformed-`xp` operator
   ([`../deformed-xp.md`](../deformed-xp.md)).
3. **A prime-driven flagship.** Geometry alone gives Weyl density, and
   arithmetic geometry over-symmetrizes to Poisson, so the fluctuations must
   come from the **primes** (the explicit formula) — the Connes–Consani–Moscovici
   operator ([`../ccm-operator.md`](../ccm-operator.md)) and the Dirac-mirror
   track ([`../dirac-mirror.md`](../dirac-mirror.md)).
