# Bibliography — the source papers this repo is built on (#76)

The primary working set of papers behind the experiments here, with stable links
so anyone can read the sources and so the local reference cache can be rebuilt if
lost. Every entry is a **public arXiv preprint** (and, where published, its
journal of record).

This file is the *tracked* parallel to a gitignored local archive: the PDFs
themselves are personal-use downloads kept under `_private/papers/` (not
redistributed — see the OSS posture note in the top-level `CLAUDE.md`), so they
don't travel with the repo. This index does, so:

- **contributors** can see exactly which papers the code and `knowledge/` notes
  rest on, even though the PDFs aren't committed; and
- the cache is **reconstructible** — each PDF is at `https://arxiv.org/pdf/<id>`,
  named `<arxiv-id>_<shortcite>_<slug>.pdf` locally.

The local archive carries its own maintainer-only manifest at
`_private/papers/INDEX.md` — the same working set with local filenames, fetch
commands, and offline "verify-against-the-PDF" pointers. The two are split on
purpose, with one canonical owner each so they don't drift into competing copies:
**this tracked file is canonical for citations and links**; `INDEX.md` is
canonical for the local-cache layout. When the working set changes, update both.

Scope: this indexes the **primary set** that's cached for offline "verify against
the PDF" work. The broader secondary literature (foundational random-matrix,
Katz–Sarnak, Bombieri–Lagarias, Montgomery–Odlyzko, etc.) is cited in the
**`Sources`** section of each individual note in `knowledge/`; this file does not
duplicate those.

The grouping mirrors the project's tracks: the Connes–Consani–Moscovici / van
Suijlekom **flagship**, the De Bruijn–Newman **warm-up**, the Sierra **`xp` /
Dirac-mirror** warm-up, and the 2026 **frontier-survey** additions.

---

## Flagship: Connes–Consani–Moscovici / van Suijlekom

- **Connes, Consani & Moscovici — *Zeta Spectral Triples*** (Nov 2025).
  [arXiv:2511.22755](https://arxiv.org/abs/2511.22755). **Primary source.** The
  finite-cutoff operator the flagship implements; its spectrum is compared to the
  zeros (forward). Pinned equation-by-equation in
  [`ccm-operator.md`](ccm-operator.md); also drives
  [`ccm-convergence-law.md`](ccm-convergence-law.md),
  [`connes-cvs-oracle.md`](connes-cvs-oracle.md), and
  [`project-framing.md`](project-framing.md). Issues #3 / #8 / #9 / #16.

- **Connes & van Suijlekom — *Quadratic Forms, Real Zeros and Echoes of the
  Spectral Action*** — Comm. Math. Phys. **406** (2025).
  [arXiv:2511.23257](https://arxiv.org/abs/2511.23257). The
  Carathéodory–Fejér self-adjointness / real-spectrum theorem and the quadratic
  form `Q(c)` the `connes-cvs` oracle implements. See
  [`ccm-operator.md`](ccm-operator.md),
  [`connes-cvs-oracle.md`](connes-cvs-oracle.md). Issue #16.

- **Connes & Consani — *Spectral triples and ζ-cycles*** — L'Enseignement
  Mathématique **69** (2023), 93–148.
  [arXiv:2106.01715](https://arxiv.org/abs/2106.01715). The spectral-triples /
  ζ-cycles framework the 2025 operator paper rests on. See
  [`ccm-operator.md`](ccm-operator.md), [`project-framing.md`](project-framing.md).

- **Connes, Consani & Moscovici — *Zeta zeros and prolate wave operators*** —
  Annals of Functional Analysis **15** (2024), no. 87.
  [arXiv:2310.18423](https://arxiv.org/abs/2310.18423). The prolate-spheroidal
  "educated guess" eigenfunction `k_λ` and the limit-control obstacle that remains
  the main gap to RH per the primary source. See [`ccm-operator.md`](ccm-operator.md),
  [`frontier-survey-2026.md`](frontier-survey-2026.md).

- **Connes — *The Riemann Hypothesis: Past, Present and a Letter Through Time***
  (Feb 2026). [arXiv:2602.04022](https://arxiv.org/abs/2602.04022). Expository
  context; the source of the first-50-zeros data at cutoff `c = 13` used as an
  oracle cross-check. See [`connes-cvs-oracle.md`](connes-cvs-oracle.md),
  [`ccm-convergence-law.md`](ccm-convergence-law.md).

## De Bruijn–Newman warm-up (#20)

- **D.H.J. Polymath — *Effective approximation of heat flow evolution of the
  Riemann ξ function, and a new upper bound for the de Bruijn–Newman constant*** —
  Research in the Mathematical Sciences **6** (2019).
  [arXiv:1904.12438](https://arxiv.org/abs/1904.12438). **Primary for #20.** The
  exact heat-flow conventions the spike implements (`H_t`, `Φ`, `H_0 = ⅛ ξ`) and
  the upper bound `Λ ≤ 0.22`. See [`debruijn-newman-flow.md`](debruijn-newman-flow.md).

- **Rodgers & Tao — *The De Bruijn–Newman constant is non-negative*** — Forum of
  Mathematics, Pi **8** (2020).
  [arXiv:1801.05914](https://arxiv.org/abs/1801.05914). The matching lower bound
  `Λ ≥ 0`; with RH ⟺ `Λ ≤ 0` this pins `Λ = 0` under RH, and explains why raising
  `t` increases rigidity (heat flow toward the picket-fence equilibrium). See
  [`debruijn-newman-flow.md`](debruijn-newman-flow.md).

## Sierra `xp` / Dirac-mirror warm-up

- **Sierra & Rodríguez-Laguna — *The H = xp model revisited and the Riemann
  zeros*** — Phys. Rev. Lett. **106** (2011), 200201.
  [arXiv:1102.5356](https://arxiv.org/abs/1102.5356). **Primary for the
  deformed-`xp` warm-up.** The self-adjoint `H = xp` quantization whose spectrum
  reproduces the zeros' **mean density** only (fluctuations left open). See
  [`deformed-xp.md`](deformed-xp.md), [`project-framing.md`](project-framing.md).
  Issues #23 / #24 / #31.

- **Sierra — *The Riemann zeros as spectrum and the Riemann hypothesis*** —
  Symmetry **11** (2019), no. 4, 494.
  [arXiv:1601.01797](https://arxiv.org/abs/1601.01797). The map of the whole
  `xp`/Riemann-zeros program: states plainly that the modified-`xp` spectra carry
  **no trace of the exact zeros**, introduces the prime-driven massless-Dirac
  model, and records the Bender–Brody–Müller operator as **non-self-adjoint**
  (out of charter). See [`deformed-xp.md`](deformed-xp.md),
  [`dirac-mirror.md`](dirac-mirror.md),
  [`frontier-survey-2026.md`](frontier-survey-2026.md).

- **Sierra — *A quantum mechanical model of the Riemann zeros*** — New J. Phys.
  **10** (2008), 033016.
  [arXiv:0712.0705](https://arxiv.org/abs/0712.0705). The 2008 precursor to the
  Dirac-mirror model (zeros as bound states in a continuum). See
  [`dirac-mirror.md`](dirac-mirror.md).

- **Sierra — *The Riemann zeros as energy levels of a Dirac fermion in a potential
  built from the prime numbers in Rindler spacetime*** — J. Phys. A: Math. Theor.
  **47** (2014), 325204.
  [arXiv:1404.4252](https://arxiv.org/abs/1404.4252). **Primary for #25.** The
  massless-Dirac prime-mirror model and the forward locator
  `M'_z(n) = Σ_{k≤n} μ(k) k^{−z}`, whose magnitude peaks at the zeros — the object
  the GPU locator (and its Dirichlet-`L` / family generalizations, #42 / #51 / #68)
  scales. See [`dirac-mirror.md`](dirac-mirror.md).

## Frontier survey, 2026 (#50)

Genuinely past the project's training cutoff; surfaced by the literature pass.

- **Groskin — *High-Precision Approximation of Riemann Zeros via the Truncated
  Weil Form*** (May 2026).
  [arXiv:2605.20224](https://arxiv.org/abs/2605.20224). The first public
  implementation of the CvS Galerkin matrix at high precision — the provenance of
  the repo's `--extra oracle` (`connes-cvs`). Convergence to the true zeros is
  still open ("we make no claim of proof"). See
  [`connes-cvs-oracle.md`](connes-cvs-oracle.md),
  [`frontier-survey-2026.md`](frontier-survey-2026.md).

- **Śliwiński — *Spectral Analysis of the `D_log^(λ,N)` Operators*** (Jan 2026).
  [arXiv:2601.12133](https://arxiv.org/abs/2601.12133). Proves the CCM
  spectrum-vs-zeros dissonance is **inverse-logarithmic** (∝ the distribution of
  primes) — the analytic backbone of the convergence-law measurement. See
  [`ccm-convergence-law.md`](ccm-convergence-law.md),
  [`frontier-survey-2026.md`](frontier-survey-2026.md). Issue #65.

- **Hateley — *A Chiral Adelic Dirac Operator and the Spectral Realization of the
  Riemann Zeros*** (Nov 2025).
  [arXiv:2511.18309](https://arxiv.org/abs/2511.18309). A forward chiral adelic
  Dirac operator on the idèle class space; zeros appear as spectral-shift-function
  discontinuities, with finite-prime truncations computable. Exploratory candidate
  **F6**: **gate cleared (forward — no ζ/zero data tunes the mass term), closed as
  won't-build (#66)** — the offered prime-built coefficient families don't
  reproduce the zeros (random synthetic Hecke data, no convergence), the matching
  is deferred to an inverse existence conjecture, and the truncated model is a
  trivial 2×2 fiber with no eigensolve/GPU surface. See
  [`frontier-survey-2026.md`](frontier-survey-2026.md). Issue #66.

## Arithmetic correlations warm-up (#84)

- **Conrey & Snaith — *Applications of the L-functions ratios conjectures*** —
  Proc. London Math. Soc. **94** (2007), 594–646.
  [arXiv:math/0509480](https://arxiv.org/abs/math/0509480). **Primary for
  #84.** Theorem 4.1 (eqs. 4.20–4.27): the full lower-order pair correlation of
  the zeros from the ratios conjecture — the modern, explicitly computable form
  of the Bogomolny–Keating result that
  `arithmetic_correlations.cs_pair_density` implements. Pinned
  equation-by-equation in
  [`arithmetic-correlations.md`](arithmetic-correlations.md).

- **Bogomolny — *Quantum and arithmetical chaos*** (Les Houches lectures,
  2003). [arXiv:nlin/0312061](https://arxiv.org/abs/nlin/0312061). The
  Bogomolny–Keating Hardy–Littlewood form of the same two-point function
  (`Φ^diag`/`Φ^off` over the twin-prime singular series) — transcribed
  independently and asserted exactly equal to CS Theorem 4.1 in the tests —
  plus the Odlyzko `10²³` deviation plots this experiment reproduces at low
  height. See [`arithmetic-correlations.md`](arithmetic-correlations.md).

## Lehmer-pair / small-gap census (#86)

- **Csordas, Smith & Varga — *Lehmer pairs of zeros, the de Bruijn–Newman
  constant Λ, and the Riemann Hypothesis*** — Constructive Approximation **10**
  (1994), 107–129. No arXiv copy. **Primary for #86.** The Lehmer-pair
  criterion `Δ²g < 4/5` and the per-pair lower bound `λ ≤ Λ` that
  `lehmer_census.csv_lambda` implements (consumed here via Stopple's
  restatement, below). See [`lehmer-census.md`](lehmer-census.md).

- **Csordas, Odlyzko, Smith & Varga — *A new Lehmer pair of zeros and a new
  lower bound for the de Bruijn–Newman constant Λ*** — Electron. Trans. Numer.
  Anal. **1** (1993), 104–111. The pair `{γ_1048449114, γ_1048449115}` at
  `t ≈ 3.8886e8` and the bound `−5.895e−9 < Λ`; the empirical pin for the
  factor-4 normalization between the γ-coordinate formula and every published
  `Λ` table (see [`lehmer-census.md`](lehmer-census.md)).

- **Stopple — *Lehmer pairs revisited*** — Experimental Mathematics **26**
  (2017), 45–53. [arXiv:1508.05870](https://arxiv.org/abs/1508.05870). The
  precise CSV definition in zeta-ordinate coordinates (eqs. 2–5) that
  `lehmer_census.csv_g`/`csv_lambda` transcribe, the section-6 data that
  reconstructs the COSV pair's `Δ ≈ 1.0857e−4`, and the `7398 / 114661`
  Lehmer-pair count on `10⁶ ≤ t ≤ 10⁶ + 6·10⁴` that the census reproduces
  forward. See [`lehmer-census.md`](lehmer-census.md).

---

*Rebuild the local cache:* fetch each PDF from `https://arxiv.org/pdf/<id>` (e.g.
`https://arxiv.org/pdf/2511.22755`) into `_private/papers/`. New additions to the
working set should be added both to the local archive's `INDEX.md` and here.
