# The Davenport–Heilbronn negative control (#85)

> Status: **delivered** (measured). The repo's first falsifiability control: the
> forward machinery run on a functional-equation function that **provably
> violates RH**. Code: `davenport_heilbronn.py`; runner:
> `scripts/run_davenport_heilbronn.py`; tests: `test_davenport_heilbronn.py`.

## Why a negative control

Every other experiment here points the forward statistics at objects believed
to satisfy RH and GUE universality — so a pipeline bug that *always reports*
"on-line / GUE" would never be caught. The Davenport–Heilbronn function f is
the canonical falsifier: it has the same surface anatomy as a Dirichlet
L-function (real Dirichlet coefficients, an exact Riemann-type functional
equation, a critical line carrying most of its zeros) but **no Euler product**,
and it has zeros **off the critical line** at computable heights. A forward
pipeline that cannot tell f from ζ is not measuring arithmetic; this note
records what each readout measures on f, and which readouts actually carry the
discriminating power.

## The object

The classical construction (Davenport–Heilbronn 1936; Titchmarsh §10.25), in
the form the code uses:

```
f(s) = sum_{n >= 1} b(n) n^{-s},      b = (1, kappa, -kappa, -1, 0)  period 5
     = [ (1 - i kappa) L(s, chi) + (1 + i kappa) L(s, chibar) ] / 2
```

with `chi` the odd complex character mod 5 (`chi(2) = i`) and `kappa` fixed by
self-duality. Writing the root number of `chi` as `eps = tau(chi)/(i sqrt 5) =
e^{2 i alpha}` (Gauss sum `tau`), the completed function

```
Lambda(s) = (5/pi)^{(s+1)/2} Gamma((s+1)/2) f(s)
```

satisfies `Lambda(s) = Lambda(1-s)` exactly when `kappa = tan(alpha)`
(`dh_kappa`). Both conditions `(1 - i kappa) eps = 1 + i kappa` and its
conjugate hold simultaneously because `eps_chibar = conj(eps_chi)` for the odd
character. **Verified:** `tan(alpha)` agrees with the classical closed form
`kappa = (sqrt(10 - 2 sqrt 5) - 2)/(sqrt 5 - 1) = 0.284079043840…` to 1e-16,
and the functional equation holds at mpmath precision (test-pinned; the
residual floors at kappa's fp64 rounding).

Since the `b(n)` are real, `Lambda` is real on the critical line, so f has a
Hardy-Z analogue `Z_f(t) = e^{i theta_5(t)} f(1/2 + it)` with
`theta_5(t) = Im log Gamma(3/4 + it/2) + (t/2) log(5/pi)` — and the smooth
zero count `N_f(t) ≈ theta_5(t)/pi + 1` (`dh_theta` / `dh_smooth_count`). The
component L-functions get the same treatment with the half-root-number
rotation `eps^{∓1/2}`, so f, `L_chi` and `L_chibar` are all scanned by the
*same* code path (`line_scan`) — the point of a control.

**Evaluator.** fp64 Euler–Maclaurin Hurwitz zeta (`hurwitz_zeta_em`):
`f(s) = 5^{-s} sum_a b(a) zeta(s, a/5)`, vectorised over the grid, valid at
any `Re s` (one pole at s = 1). At the modest heights this control needs
(`t <~ 10^3–10^4`) no Riemann–Siegel machinery is required; agreement with
mpmath is ~1e-13 relative at t = 300 (test-pinned at 1e-11), and the rotation
residual `max |Im Z|` sits at ~1e-13 — the cheap end-to-end fp64 health check.

**Forward status.** The inputs are the mod-5 characters and the algebraic
constant `kappa` — the same pure-number-theory ingredients as `dirichlet.py`.
Everything else (on-line zeros, off-line zeros, statistics, locator response)
is *computed as output* and only then compared, including against the
published off-line tables. Nothing consumes a zero.

## Readout 0 — the off-line census (the genuine RH violations)

`off_line_zeros` grids `|f|` over a `sigma > 1/2` box (fp64), takes local
minima, and polishes each candidate with mpmath `findroot`, keeping verified
zeros only (`|f(rho)| < 1e-20` at 30 digits — checked **before** rounding the
root to fp64, where the residual would be `|f'| · O(1e-14)`). Measured census
for `2 < t < 200`, `sigma > 1/2`:

| rho (computed output)                  | check |
|----------------------------------------|-------|
| 0.808517182457 + 85.699348485378 i     | matches Balanzario–Sánchez-Ortiz (0.808517 + 85.699348i) |
| 0.650830080610 + 114.163342730757 i    | |
| 0.574356050451 + 166.479305913168 i    | |
| 0.724257694627 + 176.702461242856 i    | |

Each zero here has mirrors at `1 - rho` (functional equation) and at the
conjugates, so each *representative* removes **2** zeros from the critical
line's count. The bookkeeping closes: over `(2, 200)` the on-line scan finds
122 zeros against a smooth count of 129.7 — deficit 7.7 ≈ 2 × 4 found
representatives. The deficit grows with height (~55.6 missing by t = 1000,
i.e. ~28 off-line representatives below t = 1000, density increasing with t)
and is **scan-robust** (identical counts at step 0.05 and 0.02): it is real
zeros leaving the line, not a scanning artifact.

## Readout 1 — the growth dichotomy fires on a *genuine* off-line zero

The #43 RH-by-contradiction demo exhibited Sierra's growth law with a
*planted* counterfactual zero. f supplies a real one. The discriminating
object is the partial sum of the Dirichlet **inverse** of f
(`dirichlet_inverse`: `c(1) = 1`, `c(m) = -sum_{d|m, d>1} b(d) c(m/d)` — the
analogue of `chi mu` when there is no Euler product to make the inverse
multiplicative):

```
M(n; E) = sum_{k <= n} c(k) k^{-1/2 - iE}
```

Measured log-log RMS slopes at `n_max = 10^6`:

| energy E | slope | prediction |
|---|---|---|
| 85.6993 (off-line zero, sigma_c = 0.8085) | **+0.333** | `sigma_c - 1/2 = 0.3085` |
| 50.2401 (on-line zero) | +0.103 | log-growth (≈ 0 on this measure) |
| 51.3 (generic) | +0.104 | bounded for a genuine L |

Two findings in one table:

1. **The dichotomy fires on a genuine violation**: the off-line zero's
   `n^{sigma_c - 1/2}` growth is clean (figure
   `davenport_heilbronn_growth.png`, the measured curve tracks the predicted
   guide), three times the background slope. The #43 demo's mechanism is
   confirmed on a real object, not just a planted one.
2. **The background itself is a no-Euler-product signature.** For Möbius
   weights on a genuine L-function the off-zero profile is *bounded* (slope
   ~0, the #25/#42 behaviour). Here the background sits at ~+0.10 at every
   energy because `c(n)` is not square-root-bounded — f's zeros in
   `Re s > 1` (Davenport–Heilbronn's classical theorem) force the inverse
   coefficients to grow (measured `max |c|`: 11.6 at n=1e3 → 1421 at n=1e6).

## Readout 2 — statistics: repulsion is *retained*; the count is what betrays f

Harvest to t = 1000 by the same sign-change scan for all three functions
(f: 848 zeros, `L_chi`: 904, `L_chibar`: 903; smooth count 903.6 per family):

| statistic | f | L_chi | superposition (chi ∪ chibar) | GUE | Poisson |
|---|---|---|---|---|---|
| mean folded ratio ⟨r̃⟩ | 0.6453 | 0.6383 | 0.4347 | 0.6027 | 0.3863 |
| tiny gaps (s < 0.2) | 0.35% | — | 8.9% | ~0.25% | ~18% |
| smooth-count deficit | **55.6** | ~0 | — | — | — |

The naive expectation — a linear combination of two L-functions should show
the **superposition** of two independent point processes (no repulsion across
components; the Bombieri–Hejhal picture, asymptotically) — is **not** what
modest height shows. f's on-line zeros keep full GUE-level repulsion: the
spacing histogram hugs the Wigner surmise (figure
`davenport_heilbronn_stats.png`), ⟨r̃⟩ is *above* GUE by about the same
low-height excess the genuine `L_chi` shows (0.645 vs 0.638, both
band-stable over (5,1000)), and tiny gaps are GUE-rare — nothing like the
forward-computed superposition reference (built from our own chi/chibar
harvests, density-matched by construction), which sits at ⟨r̃⟩ = 0.435 with
25× more tiny gaps.

**The control's lesson** (the mirror of #87's): *local* spacing statistics
cannot tell Davenport–Heilbronn from a genuine L-function at these heights —
universality is exactly the part of the signal that carries no arithmetic.
What betrays f in the statistics readout is **global counting, not local
spacing**: the genuine L-function's harvest matches its smooth count to <1
zero, while f is missing ~56 zeros by t = 1000 — the off-line pairs of
readout 0. A GUE-agreement result alone (the warm-up phase's headline
statistic) would have passed f; the pipeline distinguishes because it also
counts.

(If the Bombieri–Hejhal superposition regime has an onset, it is above these
heights: ⟨r̃⟩_f shows no drift toward 0.435 in any band up to t = 1000.
Pinning the onset height empirically would be a separate, larger harvest.)

## Readout 3 — the locator's response: off-line zeros become the signal

The #42 mirror locator scanned with the `1/f` inverse weights `c(k)` against
the genuine `chi mu` weights, same grid (E ∈ (2, 120), step 0.01) and
truncation (n = 1e5), threshold `0.3 log n`:

| weights | median |M| | max |M| | peaks | matched | false |
|---|---|---|---|---|---|
| `1/f` inverse | 2.05 | **87.0** | 75 | 54/64 | **21** |
| `chi mu` (L_chi) | 1.19 | 10.1 | 49 | 49/68 | **0** |

The genuine locator is clean: every peak above threshold is a real zero. The
`1/f` locator still finds most on-line zeros (54/64) but its false peaks are
not noise — they **cluster around the off-line ordinates**, with the two
dominant mounds peaking at E = 85.700 and 114.170 (the off-line zeros to
3 decimals) and rising to |M| ≈ 87 and 30, an order of magnitude above every
genuine on-line peak (figure `davenport_heilbronn_locator.png`). The Perron
mechanism behind readout 1 explains both the height (`n^{0.31}` at the zero
vs `n^{~0.1}` background) and the width (the `1/|E - t_0|` damping of a
nearby off-line zero leaks into neighbouring energies, hence a mound, not a
spike). On an object with RH violations, the locator effectively stops being
an on-line zero locator and becomes an **off-line zero detector** — a
qualitative failure signature no genuine L-function shows.

## What the control certifies

- The repo's three forward instruments (growth law, spacing statistics +
  counting, mirror locator) **jointly distinguish** a genuine L-function from
  a functional-equation lookalike with RH violations. None of them had to be
  told anything about f beyond its coefficients.
- The discriminating power is unevenly distributed, and knowing where it
  lives is the deliverable: **local spacing statistics alone are blind**
  (readout 2), exactly as #87 found on the operator side that local
  statistics can't see the zero-tracking information. Counting against the
  smooth density, the growth exponent, and the locator's mound signature are
  where the arithmetic actually shows.
- The #43 growth-law demo is upgraded from a planted counterfactual to a
  genuine, independently verified off-line zero.

## Caveats

- fp64 end-to-end: heights here (t ≤ 10^3) are far below the #55 phase
  ceiling; the rotation residual (~1e-13) is monitored per scan. Pushing the
  harvest to t ~ 10^5+ would need the Euler–Maclaurin term count rethought
  (cost grows ~ t) — and an approximate-functional-equation evaluator if it
  ever matters.
- The off-line census is a *box scan*: completeness inside the box depends on
  the fp64 grid resolving every `|f|` minimum (0.05 × 0.025 here, minima an
  order of magnitude deep — comfortable, but a coarser grid would miss
  shallow pairs). The deficit bookkeeping is the independent cross-check on
  the census's completeness.
- The superposition reference is the *finite-height empirical* union of the
  two component harvests — the right reference for a finite-height
  comparison, but not a theorem-grade stand-in for the Bombieri–Hejhal
  asymptotic statement, whose onset height this experiment deliberately does
  not claim to locate.

## Sources

- H. Davenport & H. Heilbronn, *On the zeros of certain Dirichlet series*,
  J. London Math. Soc. 11 (1936) 181–185 (and II, ibid. 307–312).
- E.C. Titchmarsh, *The Theory of the Riemann Zeta-Function* (2nd ed., Oxford
  1986), §10.25 — the period-5 coefficient form and kappa.
- R. Spira, *Some zeros of the Titchmarsh counterexample*, Math. Comp. 63
  (1994) 747–748.
- E.P. Balanzario & J. Sánchez-Ortiz, *Zeros of the Davenport–Heilbronn
  counterexample*, Math. Comp. 76 (2007) 2045–2049 — the published off-line
  zero locations our census is validated against.
- E. Bombieri & A. Ghosh, *Around the Davenport–Heilbronn function*, Russian
  Math. Surveys 66:2 (2011) — the modern study of the off-line zero
  distribution.
- E. Bombieri & D. Hejhal, *On the distribution of zeros of linear
  combinations of Euler products*, Duke Math. J. 80 (1995) 821–862 — the
  asymptotic superposition expectation for readout 2.
- G. Sierra, arXiv:1404.4252 §XII — the growth-law mechanism readout 1 reuses
  (see `dirac-mirror.md`).
