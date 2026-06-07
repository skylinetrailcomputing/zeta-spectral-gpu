# Li's criterion — a forward, computable RH probe (#52)

*Why* the Li coefficients belong here, how to compute them **without ever reading
a zero**, and what the forward sweep shows. Read
[`project-framing.md`](project-framing.md) first for the forward-vs-inverse rule.
This is the **scalar shadow** of the flagship's Weil positivity: where the CCM
operator encodes RH as `λ_min(c) ≥ 0` of a finite-cutoff quadratic form
([`ccm-operator.md`](ccm-operator.md)), Li's criterion encodes the *same* Weil
positivity as the non-negativity of a sequence of cheap numbers — a complementary,
far cheaper forward experiment with a direct conceptual line to the flagship.

## The object: Li's criterion

Li (1997): the Riemann Hypothesis holds **iff** every Li coefficient is
non-negative,

    λ_n = Σ_ρ [ 1 − (1 − 1/ρ)^n ] ≥ 0     for all n = 1, 2, 3, …,

where `ρ` runs over the nontrivial zeros. Each `λ_n` is a real number; a single
negative one would disprove RH. The numbers `λ_n` are also, equivalently, a
weighted sum of Weil's explicit-formula functional applied to a particular test
function — which is exactly why Li positivity *is* Weil positivity in scalar form.

## Why this is forward, not inverse

The discriminating test is *"does it consume the zeros as input?"* The `Σ_ρ` form
above **does** — it is the inverse formulation, and the project rule forbids it.
**We never use it.** Instead we compute the identical numbers the **forward** way,
from the Taylor expansion of `log ξ` at `s = 1` (Bombieri–Lagarias 1999):

    λ_n = (1/(n−1)!) · dⁿ/dsⁿ [ s^{n−1} log ξ(s) ]_{s=1}.

Writing `s = 1 + u` and `log ξ(1+u) = Σ_k a_k u^k`, the derivative collapses to a
finite, **zero-free** combination of the Taylor coefficients:

    λ_n = n · Σ_{j=0}^{n−1} C(n−1, j) · a_{n−j}.

The `a_k` are pure analytic data of the completed zeta
`ξ(s) = ½ s(s−1) π^{−s/2} Γ(s/2) ζ(s)`, one closed form per factor:

| factor | contribution to `a_k` |
|---|---|
| `log s` | `(−1)^{k+1} / k` |
| `−(s/2) log π` | `−(log π)/2` at `k = 1`, else `0` |
| `log Γ(s/2)` | `2^{−k} ψ^{(k−1)}(½) / k!` (polygamma at ½) |
| `log[(s−1) ζ(s)]` | `log` of the entire series whose coefficients are the **Stieltjes constants** `γ_m`, taken via a `log(1 + series)` recurrence |

So the only inputs are `π`, the polygamma values at `½`, and the Stieltjes
constants (the Laurent data of `ζ` at its pole). **No zero is ever read.** The
zeros enter only as the yardstick: the computed `λ_n` are *compared* to the
non-negativity RH predicts. Litmus: there is no inverse step, so there is nothing
to "track" a planted answer — the positivity is produced by the analytic structure
of `ξ`, not fit.

The forward computation is anchored three independent ways (the tests): the **exact
closed form** `λ_1 = 1 + γ/2 − ½ log(4π)` (which the assembly reproduces to full
precision), the published constants `λ_1…λ_5`, and an independent **Cauchy-integral**
Taylor of `log ξ` (still zero-free) that agrees to ~40+ digits.

## The forward result

Over the swept range every `λ_n` comes out **strictly positive** — consistent with
RH (a finite range, so not a proof). They climb monotonically and settle onto the
RH asymptotic growth law

    λ_n ~ (n/2)(log n + γ − 1 − log 2π)        (n → ∞)

(Keiper; Coffey; Voros): the relative deviation from this main term shrinks from
~18 at `n = 2` to ~0.08 at `n = 30` (the `λ_n` sit just above the main term — the
positive `O(√n)` remainder). `scripts/run_li_criterion.py` writes the two-panel
figure (the `λ_n` bars with the asymptotic overlaid; the relative deviation
collapsing on a log axis). A *negative* `λ_n` is the only outcome that would refute
RH; none appears, and the run reports the minimum coefficient and its index as the
forward verdict.

## Precision reality, and where the GPU actually fits

This is the repo's recurring **fp64 wall** in scalar form. Forming `λ_n` weights
the `a_k` by binomials up to `C(n−1, ·) ~ 2ⁿ` while the result is only `O(n log n)`,
so there is genuine cancellation; with the Stieltjes constants needing their own
digits, the working precision must grow with `n` (empirically `dps ≈ n + 30` keeps
tens of digits — `li_criterion.default_dps`, and `evaluate` reports a stability
residual so an under-resolved sweep is visible, never mistaken for a negative
signal). A plain float64 sweep saturates after a few dozen coefficients. So, as
CLAUDE.md warns, this is **not** "one big fp64 pass" — it is an mpmath computation.

Consequently the GPU charter angle is **not** a single deeper `ζ` Li sweep (that is
precision-bound, mpmath's job). It is the **parallel-over-family** generalisation
(Phase-2, #71, delivered below), exactly mirroring the Katz–Sarnak track
([`katz-sarnak-families.md`](katz-sarnak-families.md), #51 → batched-kernel #68).

## Phase-2: the GRH family sweep (#71)

The **Generalized** Riemann Hypothesis for a Dirichlet `L`-function `L(s, χ)` is
equivalent to the non-negativity of *its* Li coefficients. For a primitive
non-principal character `χ` mod `q`, the completed `L`-function

    Λ(s, χ) = (q/π)^{(s+a)/2} Γ((s+a)/2) L(s, χ),   a = 0 (χ even) / 1 (χ odd),

is entire of order 1 with zeros exactly the nontrivial zeros of `L(s, χ)`, so the
**same** Bombieri–Lagarias combination carries over: with
`log Λ(1+u, χ) = Σ_k a_k(χ) u^k`,

    λ_n(χ) = n · Σ_{j=0}^{n−1} C(n−1, j) · a_{n−j}(χ).

The only new input over the `ζ` case is the character. The `a_k(χ)` are assembled
factor-by-factor exactly as before, now **parity-aware**: the gamma/`(q/π)` terms use
`arg = (1+a)/2` (polygamma at `½` for even `χ`, at `1` for odd), and the
`log[(s−1)ζ]` factor is replaced by `log L(1+u, χ)` — whose Taylor coefficients are
the character's **generalized-Stieltjes data** (here taken as a forward Cauchy/Taylor
pass over the entire `L`; the tests anchor them against the explicit
`Σ_r χ(r) γ_n(r/q)` closed form). **No zero of any `L`-function is consumed.**

**Real vs complex — the verdict.** For a **complex** character `λ_n(χ)` is complex and
GRH is equivalent to `Re λ_n(χ) ≥ 0` for all `n` (Omar–Mazhouda 2007). For a **real**
(quadratic) character the zero multiset is conjugate-symmetric, so `λ_n(χ)` comes out
real — the imaginary part is `~0`, a built-in sanity check (`imag_residual`). Over the
swept families (quadratic fundamental discriminants and prime-modulus characters)
every `Re λ_n(χ)` is **strictly positive** — GRH-consistent across the family (a finite
range, so not a proof); the run reports the tightest margin and which character holds
it. The real/complex split is drawn in the family figure (the symmetry-type contrast).

**Where the GPU fits (honestly).** Each character is an independent forward
computation, so the family is embarrassingly parallel — one CUDA **block per
character** (`kernels/li_criterion_family.cu`), assembling the whole family's `λ_n` in
fp64. Following CLAUDE.md's precision rule, the split is: the precision-delicate
analytic inputs (the `log L` Taylor coefficients) are mpmath's job on the host; the GPU
does the fp64 *assembly* (the `log`-power-series recurrence, the gamma/`(q/π)` terms,
the Li combination, and the `min Re λ_n`/positivity reduction). fp64 saturates the
binomial cancellation past a few dozen coefficients, so the GPU is the **small-`n`
family producer** and `test_li_criterion_family.py` pins GPU-vs-CPU agreement there.
(The prime-family complex characters are *already* fp64-bound by their `np.exp`-built
representation, so the GPU path matches them naturally.) A shared GPU
generalized-Stieltjes kernel that would also produce the inputs on-device — for deeper
`n` and faster families — is the documented follow-up, mirroring the `#51 → #68` split.

## Reproduce

    uv run python scripts/run_li_criterion.py              # λ_1..λ_40, verdict + figure
    uv run python scripts/run_li_criterion.py --N 80 --dps 130   # deeper, more digits
    uv run python scripts/run_li_criterion.py --family quadratic --qmax 24 --N 20
    uv run python scripts/run_li_criterion.py --family prime --qmax 13 --N 16

The fast suite (`test_li_criterion.py`) pins the closed form, the published values,
the independent Cauchy cross-check, positivity, the growth law, and the structural
forward guarantee (it poisons `mpmath.zetazero` and confirms the computation still
runs — no zero consumed). `test_li_criterion_family.py` adds the family checks:
agreement with the independent generalized-Stieltjes closed form, the real-vs-complex
distinction, the family GRH verdict, and the GPU-vs-CPU small-`n` agreement.

## Sources

- **X.-J. Li**, *The positivity of a sequence of numbers and the Riemann
  Hypothesis*, J. Number Theory **65** (1997). The criterion.
- **E. Bombieri & J. Lagarias**, *Complements to Li's criterion for the Riemann
  Hypothesis*, J. Number Theory **77** (1999). The forward `log ξ` / Taylor-
  coefficient formulation used here (zeros never consumed).
- **J. Keiper**, *Power series expansions of Riemann's ξ function*, Math. Comp.
  **58** (1992). The `log ξ` Taylor coefficients and the asymptotic growth law.
- **M. Coffey**, *Toward verification of the Riemann Hypothesis: application of the
  Li criterion*, Math. Phys. Anal. Geom. **8** (2005); arXiv:1703.02844 (Li
  numerics). The published `λ_n` values and large-scale computation.
- See [`frontier-survey-2026.md`](frontier-survey-2026.md): no fresh 2023–2026 Li
  development surfaced; the forward niche taken here is the family/GRH parallel
  generalisation, not a deeper single-`ζ` sweep.
