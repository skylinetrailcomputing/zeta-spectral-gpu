# Arithmetic correlations beyond GUE — lower-order pair correlation and the prime form factor (#84)

> **Status:** implemented (`arithmetic_correlations.py` /
> `arithmetic_correlations_gpu.py` + `kernels/arithmetic_correlations.cu`,
> `scripts/run_arithmetic_correlations.py`). Formulas pinned against the source
> PDFs; the two published forms of the prediction are asserted equal in the
> tests. This note is the equation-by-equation record (the `ccm-operator.md`
> pattern).

## Why this experiment exists

Every prior statistic here checks the zeros against a **universal**
random-matrix prediction — sine-kernel pair correlation (#6), GUE spacings
(#5), Σ²/Δ₃ rigidity (#15), the unfolding-free r̃ (#35). Universality is
precisely the part of the story that carries **no arithmetic information**:
any generic chaotic system without time-reversal symmetry does the same. The
first place the zeros are *zeta* rather than generic GUE is the **lower-order
terms**, and those are predicted **entirely from the primes**. The #87
intermediate-statistics read established the mirror lesson on the operator
side (local statistics are provably blind to the system-specific content;
it lives in correlations); this experiment reads the same dichotomy on the
zero side, where the correlated part has an exact prime-side prediction.

Both readouts are forward: the predictions consume primes, Hardy–Littlewood
arithmetic, and `ζ` on the 1-line; the zeros enter only as the *output being
characterised*. Corrupt the zeros and the comparison degrades; nothing tracks
them.

## Pinned formula 1: Conrey–Snaith Theorem 4.1 (the implementation target)

Source: Conrey & Snaith, *Applications of the L-functions ratios conjectures*,
Proc. LMS **94** (2007); arXiv:math/0509480 — Theorem 4.1, eqs. (4.20)–(4.27),
verified against the PDF. Assuming the ratios conjecture, for a test function
`f` (even, decaying):

```
Σ_{γ,γ'≤T} f(γ−γ') =
  (1/(2π)²) ∫₀ᵀ ( 2π f(0) log(t/2π)
      + ∫₋ᵀᵀ f(r) [ log²(t/2π) + 2( (ζ'/ζ)'(1+ir)
                    + (t/2π)^{−ir} ζ(1−ir) ζ(1+ir) A(ir) − B(ir) ) ] dr ) dt
  + O(T^{1/2+ε})
```

with the principal value through `r = 0` and the arithmetic factors

```
A(η) = Π_p (1 − p^{−1−η}) (1 − 2/p + p^{−1−η}) / (1 − 1/p)²       (4.20/4.26)
B(η) = Σ_p ( log p / (p^{1+η} − 1) )²                              (4.21/4.27)
```

Reading off the bracket: the **pair density at separation `ε` and height `t`**
(ordered pairs per unit height per unit separation) is

```
ρ₂(ε; t) = (1/4π²) [ log²(t/2π) + 2 Re g(iε, t) ],
g(iε, t) = (ζ'/ζ)'(1+iε) + (t/2π)^{−iε} ζ(1−iε) ζ(1+iε) A(iε) − B(iε)
```

- `log²(t/2π)/4π² = d̄(t)²` is the uncorrelated plateau.
- The `ε → 0` double poles of `(ζ'/ζ)'(1+iε)` and `ζ(1−iε)ζ(1+iε)` cancel
  against each other (with the `(t/2π)^{−iε}` oscillation supplying the
  sine kernel); on the imaginary axis the leftover simple pole is purely
  imaginary, so `Re g` is finite — the PV is only formal here.
- All `t`-dependence is elementary (`log²(t/2π)` and `(t/2π)^{−iε}`), so the
  height-window integral `∫_{T₁}^{T₂} ρ₂ dt` is **closed-form**
  (`arithmetic_correlations.cs_pair_density`); no numerical `t` quadrature.
- Everything arithmetic converges fast: the `A` factors are `1 + O(1/p²)`,
  `B`'s terms are `O(log²p/p²)`; cutoff `p ≤ 1e5` leaves a relative tail
  `~1/(p_max log p_max)`, far below plotting resolution.
- `ζ(1±iε)` and `(ζ'/ζ)'(1+iε) = (ζ''ζ − ζ'²)/ζ²` come from mpmath on the
  1-line (small grids; cost is irrelevant).

## Pinned formula 2: the Bogomolny–Keating Hardy–Littlewood form

Source: Bogomolny, *Quantum and arithmetical chaos* (Les Houches lectures),
arXiv:nlin/0312061, Lecture 2 (the BK 1995/96 Nonlinearity result; verified
against the PDF). With `d̄ = log(E/2π)/2π`:

```
R₂(ε) = d̄² + R₂^{diag}(ε) + R₂^{off}(ε)
R₂^{diag}(ε) = −(1/4π²) ∂²/∂ε² log[ |ζ(1+iε)|² Φ^{diag}(ε) ]
R₂^{off}(ε)  = (1/4π²) |ζ(1+iε)|² e^{2πi d̄ ε} Φ^{off}(ε) + c.c.
Φ^{diag}(ε) = exp( 2 Σ_p Σ_{m≥1} (1−m)/(m² p^m) cos(m ε log p) )
Φ^{off}(ε)  = Π_p ( 1 − (1 − p^{iε})² / (p−1)² )
```

`Φ^{off}` is the Fourier side of the Hardy–Littlewood twin-prime singular
series; `e^{2πi d̄ ε} = (E/2π)^{iε}` is the CS oscillatory factor.

## The two are one formula (asserted in the tests)

CS say Theorem 4.1 was "originally found by Bogomolny and Keating"; the
equality is exact, not asymptotic. Term-by-term:

- **Oscillatory:** per prime, with `x = p^{−iε}`, `q = 1/p`, factoring the
  difference of squares gives `Φ^{off}_p(ε) = (1 − 2q + q/x)(1 − q/x)/(1−q)²
  = A_p(−iε)`, so `Φ^{off}(ε) = conj A(iε)` and the two `+ c.c.` halves swap.
- **Diagonal:** expanding `(p^{1+iε} − 1)^{−2} = Σ_{m≥2} (m−1) p^{−m}
  e^{−imε log p}` gives `Re B(iε) = Σ_p Σ_{m≥2} (m−1) log²p cos(mε log p)/p^m
  = −½ ∂²/∂ε² log Φ^{diag}(ε)`, while `∂²/∂ε² log|ζ(1+iε)|² =
  −2 Re (ζ'/ζ)'(1+iε)`.

`test_bk_equals_cs_exactly` holds the two independent transcriptions to
`atol 1e-10` (observed `~2e-13`) — a transcription error in either source
formula would have broken it (it caught one sign slip during development).

## Pinned formula 3: the form factor / Fourier prime peaks

The empirical statistic is the windowed Fourier transform of the ordinates,

```
S(u) = Σ_n w(τ_n) e^{i u τ_n},     w = rect or Hann taper on [T₁, T₂].
```

The explicit formula (equivalently Landau's `Σ_{0<γ≤T} x^ρ = −(T/2π) Λ(x)
+ O(log T)` under a window) predicts, for `u > 0` away from 0,

```
S(u) ≈ (1/2π) log(t_c/2π) W(−u)  −  (1/2π) Σ_{n≥2} Λ(n) n^{−1/2} W(log n − u)
```

where `W(y) = ∫ w(t) e^{−iyt} dt` is the (closed-form) window transform. Every
prime power `p^m` puts a peak of width `~2π/(T₂−T₁)` at `u = log p^m` with von
Mangoldt weight — the primes literally visible in the zeros' Fourier
statistics. Smoothing the peaks over a few widths reproduces the GUE diagonal
ramp exactly:

```
⟨|S(u)|²⟩ = (u/2π) ∫w²   (peaks of mean weight Λ²(n)/n at density du = 1/u…)
            → saturates at the plateau d̄ ∫w² at the Heisenberg frequency u = 2π d̄.
```

so the ramp/plateau (`diagonal_ramp`) and the spike sum (`prime_prediction`)
are two descriptions of the same object at different resolutions — the
arithmetic content **is** the resolved spike structure.

## Measured results (first-100k table, window `[30k, 74k]`, Hann)

- **Prime peaks:** `|S(log p^m)|²` matches the prime prediction at ratio
  `1.0000` for every prime power in `u ∈ [0.25, 4]` (n = 2 … 19). This is
  identity-level agreement — the explicit formula is exact, and the smooth
  Hann taper kills the boundary error terms — so it validates conventions and
  zero quality rather than a statistical law. The *statistical* content:
  between peaks `|S(u)|²` collapses to `~10⁻¹⁰` of the peak height, far
  **below** the GUE ramp. Below the Heisenberg frequency the zeros' Fourier
  transform is spike-dominated: the ensemble-averaged ramp only emerges after
  smoothing. (This is the dual statement of `F(α) = α` rigidity, and the
  comparison target the #44 finite-ε Dirac-mirror form factor was waiting
  for.)
- **Pair correlation:** over 63k zeros, rms deviation of the binned `R₂` from
  GUE-only is `0.019` (plateau units); the residual against Conrey–Snaith is
  `0.013` ≈ the per-bin shot noise `0.0147`, with correlation `+0.73` between
  the observed deviation and the predicted arithmetic terms — the zeros'
  departure from universality is the BK/CS prime terms, bin by bin, already at
  height `~5·10⁴`. (`data/arithmetic_form_factor.png`,
  `data/arithmetic_pair_correlation.png`.)
- **GPU:** the `zero_fourier` kernel (block-per-frequency, shared-memory
  tree reduction, `sincos`) reproduces the CPU reference to machine epsilon
  and runs ~286× faster at 63k zeros × 6000 frequencies on the RTX 3090.

## Height strategy (and the #55 caveat that dissolved)

The lower-order terms are `O(1/log t)`-sized *relative to the plateau* — they
are **largest at low height**. The cached Odlyzko tables (`first-100k`,
`first-2M`) are therefore the strong-signal regime, and the issue's
fp64-zero-harvest caveat (the #55 phase ceiling) never gets triggered: no new
zeros need harvesting for the primary readout. High-height checks (Odlyzko's
`10¹²`/`10²¹` windows, where the deviations shrink and pure universality takes
over) and a #55-harvested mid-height sweep remain natural follow-ons; both
sharpen the *height dependence* rather than the detection itself.

## What was deliberately not implemented

- **Berry's semiclassical Σ²(L)** (Berry 1985 Proc. R. Soc. A 400; Berry 1988
  Nonlinearity 1): the full oscillatory prime formula for the number variance.
  Σ² is an integral transform of `R₂`, so its arithmetic content is the same
  information this experiment already reads pointwise; the repo already uses
  its leading consequence (the #15 saturation scale `L* ≈ ln(T/2π)/π`). Pinned
  here for the record, not transcribed — add it only if a dedicated Σ²-vs-L
  panel becomes worth its own figure.
- **Family-level lower-order terms** (ratios-conjecture finite-conductor
  corrections for the #51 quadratic-Dirichlet one-level density) — real,
  separate scope.

## Forward audit

- Prediction inputs: primes (`Λ(n)`, Euler products, prime sums), `ζ(1±iε)`,
  `(ζ'/ζ)'(1+iε)`, window geometry. **No zeros.**
- Empirical inputs: zero ordinates as *data being transformed/binned* — the
  output side of the comparison only.
- The known-answer checks (peak ratios, CS residuals) compare *outputs*; they
  never feed back into any constructed object.

## Sources

- J.B. Conrey & N.C. Snaith, *Applications of the L-functions ratios
  conjectures*, Proc. LMS 94 (2007) 594; arXiv:math/0509480. Theorem 4.1,
  eqs. (4.20)–(4.27). **Primary.**
- E. Bogomolny, *Quantum and arithmetical chaos*, Les Houches lectures,
  arXiv:nlin/0312061 — the BK Hardy–Littlewood form (Lecture 2) and the
  Odlyzko 10²³ comparison plots.
- E. Bogomolny & J.P. Keating, *Random matrix theory and the Riemann zeros
  I & II*, Nonlinearity 8 (1995) 1115; 9 (1996) 911; PRL 77 (1996) 1472 — the
  original derivation (not on arXiv; cited via the two above).
- M.V. Berry, *Semiclassical theory of spectral rigidity*, Proc. R. Soc. A 400
  (1985) 229; *Semiclassical formula for the number variance of the Riemann
  zeros*, Nonlinearity 1 (1988) 399 — the Σ² side, pinned but not transcribed.
- E. Landau, *Über die Nullstellen der Zetafunktion* (1912) — the
  `Σ x^ρ = −(T/2π)Λ(x)` peak formula behind the form-factor prediction.
- A.M. Odlyzko, Math. Comp. 48 (1987) 273 and the zero tables — the empirical
  zero sets (`zeros.py` loaders) and the classical large-height confirmation.
