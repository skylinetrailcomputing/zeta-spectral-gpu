# Independent reproduction notes — the Connes–CvS / CCM zeta operator

> A self-contained field guide for anyone reproducing the finite-cutoff Weil-form
> operator whose spectrum approximates the Riemann zeros — the construction of
> **Connes–Consani–Moscovici** (*Zeta Spectral Triples*, arXiv:2511.22755) and the
> equivalent **Connes–van Suijlekom** Galerkin matrix `Q(c)` (arXiv:2511.23257).
> This note consolidates the five places we lost time, so the next person doesn't.
> It is written to stand on its own; the deeper, equation-by-equation internal
> notes are linked where you want more.
>
> **Forward, not inverse.** Everything here builds the matrix from the *primes* (a
> von Mangoldt sum cut at `k ≤ x = λ²`) plus archimedean / point-mass terms. The
> zeta zeros are consumed *nowhere* in the construction — they appear only at the
> very end as the yardstick the computed spectrum is measured against. That is the
> whole point of this repo; see [`project-framing.md`](project-framing.md).

## Who this is for

The audience is small and specific: people working in or adjacent to the Connes
program who want to *independently* recompute the §6 convergence table (or
Groskin's `connes-cvs` cell) and verify the construction for themselves. We did
exactly that, from the papers, with no shared code at the start, and the agreement
is real. If you are about to do the same, the gotchas below are the difference
between "matches to 55 digits in an afternoon" and "plateaus five orders short and
you don't know why."

## What we reproduced

Two anchors, cross-checked against each other and against the published numbers:

1. **Groskin's `connes-cvs` oracle cell** (the first public implementation of
   `Q(c)`; PyPI, MIT). The documented `c = 13, N = 80, T = 400, dps = 80` cell:

   | quantity | value |
   |---|---|
   | `λ_min` (smallest eigenvalue of `Q(c)`) | `2.5282661401965756026…e-59` |
   | `γ₁` recovered from the eigenvector | `14.1347251417346937904…` (true `t₁` to ~54 digits) |
   | `\|γ₁ − t₁\|` (first-zero error) | `1.77e-55` |

   We freeze this as a checked-in JSON fixture and diff our own assembly against
   it. See [`connes-cvs-oracle.md`](connes-cvs-oracle.md).

2. **The CCM 2025 §6 convergence table**, via our own from-scratch multiprecision
   assembly ([`ccm.py`](../src/zeta_spectral_gpu/ccm.py), `N = 120`, `dps ≈ 210`).
   First-zero error `\|eig₁ − t₁\|`, one column per cutoff:

   | `λ` (`x = λ²`) | `k = 1` | `k = 50` |
   |---|---|---|
   | `√12` | `3.41e-50` | `9.02e-2` |
   | `√13` | `2.44e-55` | `2.04e-3` |
   | `√14` | `1.07e-60` | `4.78e-6` |

   These match the source tables (pp. 26–27). The error shrinks fast with the
   cutoff `x` and grows with the zero index `k` — both behaviours matter below.

Three *independent* measurements (ours, connes-cvs, the published CCM/Connes
tables) agree at `c = 13` to within a fixed normalization factor (Gotcha 2). That
mutual agreement is the actual reproduction result.

## Gotcha 1 — the Weil combination sign (the expensive one)

**The single most costly trap.** The matrix is the Weil explicit-formula quadratic
form restricted to the truncated scaling-mode basis. It has three pieces:

- point-mass `W_{0,2}` (eq. 4.2),
- archimedean `W_ℝ` (eqs. 4.4–4.7),
- non-archimedean / primes `Σ_p W_p` (eq. 4.3).

They combine with the **signs of the explicit formula** (eqs. 3.10 / 3.13):

```
QW_λ^N = W_{0,2} − W_ℝ − Σ_p W_p        ← point-mass MINUS archimedean MINUS primes
```

It is natural — and wrong — to assemble all three with a `+`. The all-`+` form is
seductive because it still produces a near-singular matrix with a tiny eigenvalue
and a plausible-looking spectrum. The symptoms of the wrong sign:

- the minimal eigenvalue comes out **negative** (`~−1e-51`) instead of the
  Weil-positive tiny-positive (`~+1e-59`);
- the recovered spectrum converges toward the zeros **only to `~1e-50`, then
  plateaus** — about five orders of magnitude short of §6;
- nothing in the per-entry closed forms looks wrong, because nothing in them *is*
  wrong. Only the combination sign is.

The individual closed forms (§4.1–4.3 of [`ccm-operator.md`](ccm-operator.md)) are
unchanged; only the way you add them up flips. If your reproduction plateaus ~5
orders short with a negative ground-state eigenvalue, this is almost certainly it.

## Gotcha 2 — `λ_min` is not the zero error (and the ~1.2–1.3 factor)

Two numbers are easy to conflate and differ by orders of magnitude:

- **`λ_min(c)`** — the smallest eigenvalue of the Weil matrix, the *Weil-positivity
  proxy*. At `c = 13` it is `≈ 2.53e-59`, and it goes to zero **super-
  exponentially** in the cutoff.
- **`\|γ₁ − t₁\|`** — the *first-zero error*, how far the eigenvector's recovered
  zero sits from the true ordinate. At the same cell it is `≈ 1.77e-55` — several
  orders of magnitude larger than `λ_min`.

The published papers tabulate the **zero error**, not `λ_min`, so an absolute
`λ_min` is only meaningful relative to a normalization convention. The convention
anchor is the first-zero error at `c = 13`, where three independent runs land
within a factor of ~1.2–1.3 of each other:

| source | `\|γ₁\|` error at `c = 13` | vs connes-cvs |
|---|---|---|
| connes-cvs (reproduce-paper: `N=100, T=800, dps=150`) | `2.005e-55` | 1.00 |
| CCM 2025 §6 (`N=120, 200 dps`) | `2.44e-55` | ×1.22 |
| Connes 2026 first-50 data (arXiv:2602.04022) | `2.6e-55` | ×1.30 |

The spread is differing `N / T / precision` and normalization conventions, **not** a
correctness gap. Decide which quantity you are comparing *before* you trust an
absolute number, or you will chase a phantom ×1.2 "discrepancy" that is just a
convention.

## Gotcha 3 — precision is load-bearing end-to-end (fp64 corrupts, it doesn't blur)

The reported agreements run to `~1e-55…1e-60`, far below fp64's machine epsilon
(`~1e-16`). fp64 cannot even *represent* the result — but the deeper trap is *how*
it fails. Resolving the spectrum needs the near-null eigenvector `ξ` of the Weil
form, which lives at the sub-`ε_N` scale. Once `ε_N` underflows (almost
immediately), the fp64 `ξ` is **roundoff, not an imprecise answer** — so the fp64
*spectrum* is corrupted, not merely coarse. Measured at `N = 80`, first 12 zeros:

| `x` | genuine max error (mpmath) | fp64 max error | corruption `\|ν_fp64 − ν_mpmath\|` |
|---|---|---|---|
| 11 | `1.8e-24` | `1.1e+01` | `1.1e+01` |
| 13 | `3.9e-34` | `1.4e+01` | `1.4e+01` |
| 15 | `4.0e-44` | `2.4e+01` | `2.4e+01` |

The fp64 "error" is *entirely* `ξ`-corruption. Consequences for a reproducer:

- **Run the whole pipeline — fill *and* eigensolve — in extended precision.** The
  source says "easily performed using 200 digits"; take that literally and
  end-to-end. We use `dps ≈ 210` (mpmath; `gmpy2` backend for speed).
- **Precision must grow with the cutoff depth.** Rough budget:

  | precision | resolves cutoff up to |
  |---|---|
  | fp64 (`~16` digits) | `x ≈ 5–9` |
  | `dps = 110` | `x ≈ 30–50` |
  | `dps = 500–1000` | `x = 100` (Groskin's regime) |

- **A double-double (`~32`-digit) fill is *not* a shortcut.** We spiked this and
  it is a clean negative: ~99% of the fp64 matrix error is in the special-function
  *coefficients*, not the fill arithmetic (the near-band divided-difference
  subtraction is already exact by Sterbenz's lemma). `~32` digits is below the
  `~80` needed at `c = 13`, and an fp64 eigensolve can't consume even the 15 digits
  a careful fill already has. The real lever is high-precision coefficients
  **paired with** an extended-precision eigensolve — not a cleverer fill. Details:
  [`ccm-fill-precision.md`](ccm-fill-precision.md).

So: the GPU's honest role is the dense special-function *fill* (validated to
`~1e-12` in fp64) and the many-cutoff sweep. The delicate eigensolve stays in
extended precision on CPU. Framing this as "one big fp64 eigensolve" is the trap.

## Gotcha 4 — the eigensolve recipe that makes it tractable

Profiling surprises people: the eigensolve is **not** the cost driver. The
archimedean fill — `h_+(τ) = Re ψ(¼ + iτ/2) − log π`, i.e. millions of
high-precision complex-digamma evaluations — is ~80% of wall time; the eigensolve
is ~17%. Three moves that made our flagship ~6× faster overall, all worth copying:

1. **Factor once, inverse-iterate.** `QW` is positive with a single tiny eigenvalue
   and the rest `O(1)`, so the spectral gap is enormous and inverse iteration
   (`x ← A⁻¹x`) converges to the near-null direction in essentially one step. LU-
   factor the matrix *once* and reuse the factorization across all iterations
   rather than re-solving.
2. **Parity-reduce to the even block.** The matrix commutes with the `ℤ/2` parity
   grading, and the minimal eigenvector is even. Restricting to the even subspace
   is the same eigenproblem at half the dimension — the `O(N³)` LU shrinks by
   `((N+1)/(2N+1))³ ≈ 1/8`. As a bonus the result is *exactly* even by construction
   (no symmetrization, no odd contamination to project out; the parity residual is
   identically zero).
3. **Fuse and hoist the special-function fill.** The three archimedean integrals
   per mode share the *same* `₂F₁(1, z_k, z_k+1, e^{-2L})` and `digamma(z_k)`;
   compute each once per mode instead of 3× / 2×, and hoist the cutoff-only factors
   (`e^{-2L}`, `e^{-L/2}`, `digamma(¼)`, the `M`-constant) out of the assembly loop.

See [`connes-cvs-oracle.md`](connes-cvs-oracle.md) (profiling) and the
`smallest_even_eigenvector` / `assemble_weil_matrix` docstrings in
[`ccm.py`](../src/zeta_spectral_gpu/ccm.py).

## Gotcha 5 — small construction traps

A handful of one-line corrections we made against an early reading of the source:

- **The prime term has no separate `F(p^{-m})`.** The clean computational form is
  `Σ_{1<k≤λ²} Λ(k) k^{-1/2} q(U_n,U_m)(log k)` — one kernel evaluation per prime
  power. The reflection symmetry is *already inside* the even kernel `q`; there is
  no extra `(F(p^m) + F(p^{-m}))` factor to add.
- **The diagonal archimedean entry is delicate — cross-check it against direct
  quadrature** of the master formula (4.4). The closed form reduces to integrals
  (4.6)/(4.7) plus an `L`-only constant; a tanh-sinh quadrature at a few extra
  digits is the robust sanity check while you debug.
- **Eq. (4.6) carries a trigamma `ψ⁽¹⁾`** (the `x·cos` integral). It is easy to drop
  if you transcribe from a digamma-only reading.
- **The diagonal `L`-only constant shifts every eigenvalue but no eigenvector.** It
  adds `c·I` to the matrix, so it moves `ε_N` but leaves `ξ` (hence the extracted
  spectrum) untouched. Include it for fidelity — so the minimal eigenvalue is the
  genuine `ε_N` and not a shifted impostor — but know it is harmless to the zeros.
- **The boundary-evaluation functional `δ_N` carries unit mode coefficients**,
  `δ_N(x) = Σ_{|n|≤N} e^{2πinx/L}`. Its overall constant is immaterial because the
  normalization `δ_N(ξ) = 1` fixes the scale of `ξ`.

## Where the code is

| you want… | look at |
|---|---|
| the verified, equation-by-equation operator spec | [`ccm-operator.md`](ccm-operator.md) |
| the CPU multiprecision reference implementation | [`ccm.py`](../src/zeta_spectral_gpu/ccm.py) |
| the `connes-cvs` oracle + frozen fixture | [`connes-cvs-oracle.md`](connes-cvs-oracle.md), [`oracle.py`](../src/zeta_spectral_gpu/oracle.py) |
| reproduce the §6 table yourself | `uv run python scripts/run_ccm.py` |
| reproduce / refresh the oracle cell | `uv sync --extra oracle && uv run python scripts/run_connes_cvs_oracle.py` |
| the precision anatomy / why dd-fill is a no | [`ccm-fill-precision.md`](ccm-fill-precision.md) |
| how the spectrum approaches the zeros (the law) | [`ccm-convergence-law.md`](ccm-convergence-law.md) |
| the prime-cutoff rigidity / universality read | [`ccm-universality.md`](ccm-universality.md) |

## On sharing this — deferred to the maintainer

This note exists partly to be *shareable* with the Connes/CvS camp (Groskin and/or
the CCM group): the independent reproduction plus these gotchas is the kind of
thing the next reproducer would genuinely use. But **any actual outreach is a
maintainer decision, not an automated one** — opening an issue/PR on the
`connes-cvs` repo, emailing, or otherwise contacting an author is outward-facing
and hard to retract, so it waits for an explicit go-ahead.

Options, when/if that happens (no action taken here):

- a friendly heads-up issue on the `connes-cvs` repo linking this note (lowest
  friction; Groskin's repo already invites cross-checks);
- a short standalone gist / write-up if a non-repo-specific link is preferable;
- nothing — the note still earns its place as in-repo institutional memory for the
  next person who reproduces this, internal or external.

## Sources

- **arXiv:2511.22755** — Connes, Consani, Moscovici, *Zeta Spectral Triples*
  (27 Nov 2025). The operator, §6 convergence tables. Primary; see
  [`ccm-operator.md`](ccm-operator.md).
- **arXiv:2511.23257** — Connes & van Suijlekom, *Quadratic Forms, Real Zeros and
  Echoes of the Spectral Action*. The Galerkin matrix `Q(c)` connes-cvs implements.
- **arXiv:2605.20224** — Groskin, *High-Precision Approximation of Riemann Zeros
  via the Truncated Weil Form* (2026). The `connes-cvs` implementation; the
  super-exponential low-zero regime. Repo `github.com/akivag613/connes-cvs-`;
  Zenodo DOI 10.5281/zenodo.19546514.
- **arXiv:2602.04022** — Connes, *The Riemann Hypothesis: Past, Present and a Letter
  Through Time*. First-50-zeros data at `c = 13` (the convention table above).
- **arXiv:2601.12133** — Śliwiński, *Spectral Analysis of the `D_log^(λ,N)`
  Operators* (2026). The convergence-law context; see
  [`ccm-convergence-law.md`](ccm-convergence-law.md).
