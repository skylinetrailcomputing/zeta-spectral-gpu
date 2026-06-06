# Project framing: forward problems, never the inverse-spectral trap

## The discriminating principle

Throwing a GPU at "search for the operator whose spectrum is the Riemann zeros"
is a fool's errand in one specific, provable way, and a genuinely useful tool in
several nearby ways. The line between them:

> **Forward (valuable):** take a *structurally derived* operator and ask
> "does its spectrum converge to the zeros?"
> **Inverse (tautological):** take the *zeros* and ask "what operator fits them?"

The inverse problem **always succeeds** and teaches nothing: infinitely many
potentials reproduce any finite target spectrum to arbitrary accuracy, so the
fit merely consumes data you already had. The cautionary precedent is
**Wu–Sprung (1990s)** — a fractal potential whose energy levels reproduce the
zeros, a curiosity precisely because the construction *uses the zeros as input*.
A GPU sweep over potentials is just a faster Wu–Sprung.

**The test any experiment in this repo must pass:** *does it consume the zeros
as input?* If yes, it is curve-fitting and does not belong here.

## Where GPU numerics genuinely earns its keep (all forward)

1. **Statistical / universality experiments** — spacing distributions, pair
   correlations, GUE agreement, the De Bruijn–Newman flow on large zero sets or
   candidate spectra. Well-posed, falsifiable, embarrassingly parallel. **This is
   the warm-up phase** and the current scaffold target; it extends the CPU work
   in `wedgetrigfunctions202601`'s `gue_spacing.py` and scales it.
2. **Characterising *derived* deformed-`xp` models** (Sierra's
   `xp(1 + l^2/x^2)`-type Hamiltonians) — computing the spectra of *principled*
   candidates, not inventing one to fit.
3. **Connes–Consani–Moscovici spectral-triple operators** — a rank-one
   perturbation of a scaling operator whose matrix is the Weil explicit-formula
   quadratic form; its prime content is the von-Mangoldt/Euler sum over primes
   **`p ≤ x = λ²`**. Finite-dimensional real-symmetric matrices whose spectra are
   conjectured to converge to the low zeros as the cutoff grows. Built from
   *primes, not zeros*, so watching the zeros emerge is informative. **This is
   the flagship phase.**

## Roadmap

- **Warm-up (now):** GPU spacing/pair-correlation/universality at scale. The
  zeros appear only as an output being characterised.
- **Flagship (after a careful read of the primary source):** reimplement the
  Connes–Consani–Moscovici finite-cutoff operators and study spectral convergence
  as the prime cutoff scales. The operator definition must come from
  **arXiv:2511.22755** (Connes, Consani & Moscovici, *Zeta Spectral Triples*, Nov
  2025) and the 2023 *Enseign. Math.* framework paper (Connes & Consani,
  arXiv:2106.01715) — **not** from any secondary note. The reality-of-spectrum
  ingredient is the Carathéodory–Fejér theorem (Connes & van Suijlekom,
  arXiv:2511.23257). Pinning the construction exactly is the first task of that
  phase.

## Why not "one giant fp64 eigensolve"

Matching *low* zeta zeros to high accuracy can require precision beyond fp64,
which neither consumer GPUs nor cuSOLVER provide natively. Concretely: the
primary source runs at **~200-digit precision** (matrix dimension 241, N=120) and
reports per-zero errors down to ~**1e-55** — fp64's ~16 digits cannot even
represent that agreement. So for the flagship, the GPU's job is the
special-function matrix fill and the many-cutoff (λ) sweep; the extended-precision
eigensolve stays on CPU (`mpmath`/Arb/MPFR). The GPU's leverage is parallel
statistics and operator *assembly*, not the delicate eigenvalue convergence. See
the precision note in `CLAUDE.md` and `README.md`.

## Source

Distilled from §10 of the design note `harmonic-functions-to-zeta.md` (the
"Numerical approaches: forward problems vs. the inverse-spectral trap" section).
That note is the maintainer's design brief; this file is the public-safe
distillation.
