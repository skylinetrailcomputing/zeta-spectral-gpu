# The Connes–Consani–Moscovici operator — verified, code-ready spec

> The finite-cutoff operator the **flagship phase** implements. Every formula
> below was verified equation-by-equation against the primary source PDF of
> **arXiv:2511.22755** (Connes, Consani & Moscovici, *Zeta Spectral Triples*,
> 27 Nov 2025); equation numbers `(n.m)` refer to that paper. This pins the
> construction so the CPU multiprecision reference (#8) and the GPU assembly /
> λ-sweep (#9) implement against the source, not a paraphrase.
>
> **Forward, not inverse.** The matrix is built from the *primes* (a von
> Mangoldt sum cut at `p ≤ x = λ²`); the zeta zeros appear *only* as the thing
> the spectrum is compared against. Nothing here consumes the zeros as input —
> see [`project-framing.md`](project-framing.md).

## 0. Notation

| Symbol | Meaning |
|---|---|
| `λ > 1` | the cutoff parameter; the prime cutoff is `x = λ²` |
| `L = 2 log λ` | log-length of the interval `[λ⁻¹, λ]`; note `e^L = λ² = x` |
| `N` | truncation; the matrix has dimension `2N+1` (headline `N = 120`) |
| `γ` | Euler–Mascheroni constant |
| `ψ`, `ψ⁽¹⁾` | digamma `Γ'/Γ` and trigamma (its derivative) |
| `₂F₁(a,b;c;z)` | Gauss hypergeometric function |
| `Φ(z,2,a) = Σ_{k≥0} z^k/(a+k)²` | Hurwitz–Lerch transcendent (`s = 2`) |
| `Λ(k)` | von Mangoldt: `log p` if `k = p^m`, else `0` |

## 1. Hilbert space, scaling operator, truncation

Work on `L²([λ⁻¹, λ], du/u)` with the self-adjoint scaling operator

    D_log = −i·u·∂_u = −i·∂_{log u}        (periodic boundary conditions)

Its eigenbasis is the Fourier basis on the log-interval of length `L = 2 log λ`.
With `U_n(x) = L^{−1/2} e^{2πinx/L}` on `[0, L]` (3.21) and the isometry
`κ(f)(u) = f(log(λu))` (Prop. 3.2, 3.17), set `V_n = κ(U_n)`. Then

    D_log V_n = (2πn/L) V_n,     L = 2 log λ.

Truncate to `|n| ≤ N` ⇒ the space `E_N`, dimension `2N+1`. These are the
`2N+1` eigenfunctions of smallest `|eigenvalue|` (`≤ Nπ/log λ`).

## 2. The Weil quadratic form and where the primes enter

The matrix that is diagonalised is the **Weil explicit-formula quadratic form**
`QW_λ` restricted to `E_N`. In the `♯`/`F` variables (`F = Δ^{1/2}f`,
`F(x) = x^{1/2} f(x)`) the explicit formula (3.2) splits `QW` into three
distribution types. They combine with the **signs of the explicit formula**
(eqs. 3.10/3.13): `Ψ♯ = W_{0,2}^♯ − W_ℝ^♯ − Σ_p W_p^♯` — point-mass *minus*
archimedean *minus* primes (see §4):

- **point-mass** `W_{0,2}` — eq. (3.14),
- **archimedean** `W_ℝ` — eq. (3.15),
- **non-archimedean (primes)** `W_p` — eq. (3.16): `W_p^♯(F) = (log p) Σ_{m≥1}
  p^{−m/2} F(p^m)`.

A matrix entry is `QW_λ(V_n, V_m) = Ψ^♯(F)` with `F(x) = q(U_n,U_m)(log x)`
(Prop. 3.2, eq. 3.18). Because the test functions are supported on `[λ⁻¹, λ]`,
the kernel `q(U_n,U_m)` is supported on `[−L, L]`, so only arguments
`log k ≤ L`, i.e. **prime powers `k = p^m ≤ e^L = λ²`**, survive. Hence the
prime cutoff `x = λ²`.

## 3. The assembly primitive: `q(U_n, U_m)` (Lemma 2.3)

Every entry reduces to the **even** quadratic-form kernel (verified, Lemma 2.3,
eqs. 2.7–2.10), for `y ∈ [0, L]` and extended evenly to `[−L, L]`:

    q(U_m, U_n)(y) = ( sin(2πmy/L) − sin(2πny/L) ) / ( π(n − m) )     for m ≠ n
    q(U_n, U_n)(y) = 2 (1 − y/L) cos(2πny/L)                          for m = n

Note `q(U_m,U_n)(0) = 0` for `m ≠ n` and `q(U_n,U_n)(0) = 2`.

## 4. Matrix entries

The full entry combines the three contributions below **with the explicit-formula
signs** (eqs. 3.10, 3.13 — point-mass *minus* archimedean *minus* primes):
`(QW_λ^N)_{n,m} = W_{0,2}(V_n,V_m) − W_ℝ(V_n,V_m) − Σ_p W_p(V_n,V_m)`.

> ⚠️ **Sign correction (verified against the PDF, eqs. 3.10/3.13).** An earlier
> revision of this note wrote this as a *sum* of all three. That is wrong: with
> the all-`+` combination the operator is the wrong sign on `W_ℝ`/`Σ_p`, its
> minimal eigenvalue comes out negative (`~−1e−51` instead of the Weil-positive
> `~+1e−59`), and the spectrum converges to the zeros only to `~1e−50` before
> plateauing — five orders short of §6. With the subtraction the minimal
> eigenvalue is tiny-positive and the first zeros match §6 (`~1e−55`). The
> individual closed forms in §4.1–4.3 are unchanged; only the combination sign
> was wrong.

### 4.1 Point-mass `W_{0,2}` — Lemma 4.1 (closed form, rank-one)

    W_{0,2}(V_n, V_m) = 32 L sinh²(L/4) · (L² − 16π² n m)
                        ───────────────────────────────────────
                        (L² + 16π² m²)(L² + 16π² n²)

Verified verbatim against eq. (4.2). The paper notes this term contributes a
**rank-one** matrix.

### 4.2 Archimedean `W_ℝ` — Prop. 4.2 (eqs. 4.4–4.7)

**Master formula (4.4)**, with `ω(x) = q(U_n,U_m)(x)` and the weight
`ρ(x) = e^{x/2} / (e^x − e^{−x})`:

    W_ℝ(V_n, V_m) = (ω(0)/2) · ( γ + log(4π (e^L − 1)/(e^L + 1)) )
                  + ∫₀^L [ e^{x/2} ω(x) − ω(0) ] / (e^x − e^{−x}) dx

The single integral is well-defined: the numerator → 0 as `x → 0`. An
implementation may evaluate it by **direct high-precision quadrature** (simplest,
robust) or via the three closed-form primitives below (fast: the series
parameter is `e^{−2L} < 1`, so convergence is geometric — "for `L` of order 10"
only ~10 terms are needed).

**Primitives (4.5)–(4.7)** — verified exactly, including the Re/Im parts:

    (4.5)  ∫₀^L sin(2πnx/L) ρ(x) dx
           = e^{−L/2} Im[ (2L/(L + 4πin)) ₂F₁(1, πin/L + ¼; πin/L + 5/4; e^{−2L}) ]
             + ½ Im[ ψ(πin/L + ¼) ]

    (4.6)  ∫₀^L x cos(2πnx/L) ρ(x) dx
           = −L e^{−L/2} Im[ (2L/(4πn − iL)) ₂F₁(1, ¼ + inπ/L; 5/4 + inπ/L; e^{−2L}) ]
             − (e^{−L/2}/4) Re[ Φ(e^{−2L}, 2, iπn/L + ¼) ]
             + ¼ Re[ ψ⁽¹⁾(πin/L + ¼) ]

    (4.7)  ∫₀^L (cos(2πnx/L) − 1) ρ(x) dx
           = −e^{−L/2} Re[ (2L/(L + 4πin)) ₂F₁(1, πin/L + ¼; πin/L + 5/4; e^{−2L}) ]
             + 2 e^{−L/2} ₂F₁(¼, 1; 5/4; e^{−2L})
             − ½ Re[ ψ(πin/L + ¼) − ψ(¼) ]

The `₂F₁` parameters are `(1, πin/L + ¼; πin/L + 5/4; e^{−2L})` (so `c − b = 1`);
the sin-integral takes the **imaginary** part, the (cos−1)-integral the **real**
part. (Source proof: substitute `y = 2πx/L`, `a = L/2π`, and expand
`ρ = Σ_{k≥0} e^{b(k)y}`, `b(k) = −a(1 + 4k)/2`, which produces the `e^{−2L}`
series.)

**Off-diagonal (`n ≠ m`), worked closed form.** Here `ω(0) = 0`, so

    W_ℝ(V_n, V_m) = ( J(m) − J(n) ) / ( π(n − m) ),   J(k) := integral (4.5).

**Diagonal (`n = m`).** Substitute `ω(x) = 2(1 − x/L) cos(2πnx/L)`, `ω(0) = 2`
into (4.4); it reduces to integrals (4.6)/(4.7) plus an `L`-only constant. Given
how delicate the diagonal is, **cross-check it against direct quadrature of
(4.4)** during implementation.

### 4.3 Prime term `Σ_p W_p` — eq. (4.3), cutoff `k ≤ λ²`

The single clean computational form (verified, eq. 4.3):

    Σ_p W_p(V_n, V_m) = Σ_{1 < k ≤ exp(L)} Λ(k) · k^{−1/2} · q(U_n, U_m)(log k)

with `exp(L) = λ² = x`. Only prime powers `k = p^m ≤ λ²` contribute (`Λ(k) ≠ 0`).
For the headline `x = 13`: `k ∈ {2,3,4,5,7,8,9,11,13}`. The reflection symmetry
is already inside the even kernel `q` — there is no separate `F(p^{−m})` term
(see §9).

## 5. Matrix structure & the even-simple condition

`QW_λ^N` is **real symmetric**, dimension `2N+1`, and commutes with the `ℤ/2`
parity grading `γ` (`γ V_j = V_{−j}`). Its structure (Lemma 5.1) is
**divided-difference / Hankel-like, not plain Toeplitz**:

    τ_{i,i} = a_i,    τ_{i,j} = (b_i − b_j)/(i − j)  (i ≠ j),   i,j ∈ {−N,…,N}

with `a_{−j} = a_j` (even) and `b_{−j} = −b_j` (odd), where (from §5)

    b_n = −(1/π) ∫₀^L sin(2πny/L) 𝒟(y) dy,
    a_n =  2     ∫₀^L (1 − y/L) cos(2πny/L) 𝒟(y) dy,    𝒟 = log_λ(Ψ^♯).

(The Toeplitz / Carathéodory–Fejér structure that *forces* real spectrum enters
after a transform, via arXiv:2511.23257.)

**Even-simple (Def. 5.3):** the smallest eigenvalue is simple and its
eigenvector `ξ` is even (`γξ = ξ`). This is observed numerically, not proved for
general `N, λ` — see §8.

## 6. The rank-one perturbation & spectrum — Theorem 1.1

Let `ε_N` be the smallest eigenvalue of `QW_λ^N` (assumed simple), `ξ` its
(even) eigenvector, normalised by `δ_N(ξ) = 1`. Here `δ_N` is the **Dirichlet
kernel** (§5.3, eqs. 5.8–5.9), the equal-weight sum of all `2N+1` modes,
representing boundary evaluation:

    δ_N(x) = Σ_{n=−N}^{N} e^{2πinx/L} = sin(π(2N+1)x/L) / sin(πx/L),   δ_N(0) = 2N+1,

acting as a Dirac delta via `(1/L) ∫₀^L δ_N(x) f(x) dx → f(0)` (Lemma 5.5). Then:

    D_log^{(λ,N)} = D_log^{(λ)} − |D_log^{(λ)} ξ⟩⟨δ_N|

is self-adjoint in the form-corrected inner product on `E_N/ℂξ`, and

    det_reg( D_log^{(λ,N)} − z ) = −i λ^{−iz} ξ̂(z),

with `ξ̂` entire and **all its zeros real** — they *are* the spectrum of
`D_log^{(λ,N)}`. These eigenvalues are the zeta-zero approximations.

## 7. Numerical setup & expected accuracy (§6)

- **Dimension:** `N = 120` ⇒ `2N+1 = 241`, real symmetric.
- **Cutoffs:** `λ ∈ {√12, √13, √14}` (`x = 12,13,14`); plus a `λ = 3` (`x = 9`)
  case in Fig. 1. The headline uses primes/prime-powers `≤ 13`.
- **Precision:** the source states the computations are *"easily performed using
  200 digits accuracy"* — for **both** the special-function fill and the
  eigensolve.

Expected per-zero error `|eig_k − t_k|` (upper bounds, `N = 120`), first/last of
the first 50, matching the source tables (pp. 26–27):

| λ      | k = 1        | k = 50       |
|--------|--------------|--------------|
| √12    | 3.41e−50     | 9.02e−2      |
| √13    | 2.44e−55     | 2.04e−3      |
| √14    | 1.07e−60     | 4.78e−6      |

Error grows with index `k` and shrinks fast with `λ` (i.e. with `x`). Regularised
determinants converge (`N → ∞`) to Riemann's `Ξ` (up to an `e^{a+ibs}` factor);
the joint limit `N, λ → ∞` is demonstrated numerically, not proved (§7).

## 8. Implementation recipe & precision reality

1. Choose `N = 120`, `λ ∈ {√12, √13, √14}` (`L = 2 log λ`); primes powers
   `k ≤ x = λ²`.
2. Reference zeros `t_k` of `ζ(½ + it)` to ≥200 digits (mpmath / Odlyzko).
3. Build the `241×241` real-symmetric `QW_λ^N = W_{0,2} + W_ℝ + Σ_p W_p`
   (§4) at ~200-digit precision.
4. Diagonalise (extended precision); take smallest eigenvalue `ε_N` and its
   eigenvector `ξ`; **verify simple & even** (`γξ = ξ`); normalise `δ_N(ξ) = 1`.
5. Form `D_log^{(λ,N)} = diag(2πn/L) − |D_log^{(λ)} ξ⟩⟨δ_N|` (rank-one update of
   a diagonal); its spectrum = real zeros of `ξ̂(z)`. Cross-check via the
   `det_reg` formula.
6. Plot `|eig_k − t_k|` (log scale) vs `k = 1…50`, one curve per `λ`; compare to
   the §7 table.

**Why not one fp64 eigensolve.** The reported agreements reach `~1e-55…1e-60` —
far below fp64's machine epsilon (`~1e-16`), so fp64 cannot even *represent* the
result. This is an inference, but an unavoidable one: the source's explicit
"200 digits" is needed end-to-end. The GPU's role is therefore the dense
special-function fill and the many-`λ` sweep; the delicate eigensolve stays in
extended precision on CPU (mpmath / Arb / MPFR). See `CLAUDE.md` precision note.

**Open mathematical gap (not a coding issue).** Reality of the spectrum is
*forced* at finite `N` via the Carathéodory–Fejér extension (arXiv:2511.23257);
the even-simple hypothesis is numerically observed, not proved for general
`N, λ`. The real obstacle to RH is limit control — showing the minimal
eigenvector is approximated by the prolate-spheroidal `k_λ` of arXiv:2310.18423
as `N, λ → ∞`. The numerics here are forward evidence, not a proof.

## 9. Corrections to the earlier internal sketch

While verifying against the PDF, these points in the pre-existing maintainer
notes were corrected:

- **Combination sign (caught during the #8 numerics).** The three contributions
  combine as `W_{0,2} − W_ℝ − Σ_p W_p` (eqs. 3.10/3.13), *not* as an all-`+` sum.
  The all-`+` version plateaus five orders short of §6 and gives a negative
  minimal eigenvalue; the subtraction reproduces §6. See the box in §4.
- **Prime term has no `F(p^{−m})`.** The clean computational form is eq. (4.3),
  `Σ_{k≤λ²} Λ(k) k^{−1/2} q(U_n,U_m)(log k)` — one `q`-evaluation per prime
  power. The reflection is already in the even kernel `q`; there is no separate
  `(F(p^m) + F(p^{−m}))` factor.
- **`δ_N` carries unit mode coefficients**, `δ_N(x) = Σ_{|n|≤N} e^{2πinx/L}`
  (= `L^{1/2} Σ U_n`), paired via `(1/L)∫`. The constant differs from an earlier
  `L^{−1/2} Σ V_n` rendering; it is immaterial to the result because `δ_N(ξ) = 1`
  fixes the scale of `ξ`.
- **Eq. (4.6) contains a trigamma `ψ⁽¹⁾`** (the `x·cos` integral), absent from
  the earlier sketch.

The eq. (4.5) Re/Im assignment in the earlier sketch (**imaginary** part on the
`₂F₁`/`ψ` terms) is correct, now confirmed against the typeset source.

## 10. Sources

All public arXiv preprints; cited by equation number above.

- **arXiv:2511.22755** — Connes, Consani, Moscovici, *Zeta Spectral Triples*
  (27 Nov 2025). **Primary.** Operator definition: eqs. (3.2), (3.14)–(3.18),
  Lemma 2.3, Lemma 4.1 (4.2), Prop. 4.2 (4.3)–(4.7), Lemma 5.1, Def. 5.3,
  Theorem 1.1, §5.3 (5.8)–(5.10), §6 tables, §7 outlook.
- **arXiv:2511.23257** — Connes & van Suijlekom, *Quadratic Forms, Real Zeros
  and Echoes of the Spectral Action* (CMP 406, 2025). The Carathéodory–Fejér /
  real-spectrum theorem.
- **arXiv:2106.01715** — Connes & Consani, *Spectral triples and ζ-cycles*
  (Enseign. Math. 69, 2023). Framework.
- **arXiv:2310.18423** — Connes, Consani, Moscovici, *Zeta zeros and prolate
  wave operators* (Ann. Funct. Anal. 15, 2024). The prolate `k_λ` and the
  limit-control obstacle.
