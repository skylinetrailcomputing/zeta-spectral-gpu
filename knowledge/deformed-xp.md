# Sierra's deformed-`xp` operator — mean density without the fluctuations (#23/#24/#31)

*Why* a geometrically deformed `xp` operator belongs here, what its spectrum does
(and conspicuously does **not**) reproduce, and how the CPU reference and GPU
eigensolve split the work. Read [`project-framing.md`](project-framing.md) first
for the forward-vs-inverse rule; this is the first entry in the **xp-Hamiltonian
track** (the forward stretch contrast to the flagship).

## Why this is forward, not inverse

The discriminating test is *"does it consume the zeros as input?"* No. The
operator is built from a **geometric deformation of `xp`** — neither the primes
nor the zeros enter its definition. We diagonalise a *principled* candidate and
*compare* its spectrum to the zeros afterward. We never invent or tune one to fit
(that would be the Wu–Sprung trap).

## The object

Berry & Keating's `H = xp` reproduces the *average* Riemann zeros semiclassically
(its Weyl count matches the Riemann–von Mangoldt `N̄`), but its classical orbits
are open (non-periodic), so it cannot carry the discrete spectrum. Sierra &
Rodríguez-Laguna close the orbits with a geometric deformation,

    H = x (p + ℓ_p² / p),      x ≥ ℓ_x,   p ∈ ℝ                         (eq. 3)

whose normal-ordered quantization is a self-adjoint operator on the half-line.
In position space (Sierra eq. 10, ħ = 1):

    H ψ(x) = −i (x ψ'(x) + ½ ψ(x)) − i ℓ_p² √x ∫ₓ^∞ √y ψ(y) dy.

The boundary condition at `x = ℓ_x` quantizes the spectrum; the eigenvalues are
the real roots of a **Bessel-`K` secular equation** (eq. 14). There is also an
`x↔p`-symmetric sibling, Berry & Keating's `H_II = (x + ℓ_x²/x)(p + ℓ_p²/p)` — see
*The symmetric sibling* below. It reaches the same *average-yes, fluctuations-no*
verdict, but only **semiclassically**: unlike `H_I` it has no closed-form secular
equation (#59).

## What it teaches: mean yes, fluctuations no

The deformed-`xp` spectrum reproduces the **smooth** counting function of the
zeros (the Riemann–von Mangoldt mean density) — but its **fluctuations are wrong**.
The universality read (#24) finds the levels sit close to a rigid **picket fence**
(`⟨r̃⟩ ≫ 0.603`, `var(s) ≪ 0.178`), *not* the zeros' GUE statistics. This is the
clean, forward demonstration of an important negative: **matching the mean density
is necessary but nowhere near sufficient.** A geometric deformation can fix the
average but not the level repulsion — for that you need the primes (the
Dirac-mirror track, [`dirac-mirror.md`](dirac-mirror.md)) or the explicit-formula
quadratic form (the flagship).

This is the same picket-*ward* tendency seen elsewhere in the repo at small scale
(the flagship's small-`x` rigidity, [`ccm-universality.md`](ccm-universality.md);
the zeros' own sub-GUE saturation at long range, #15) — but here it is the whole
story, not a finite-size artefact.

## The symmetric sibling `H_II` — semiclassical only (#59)

The object above is `H_I`. Its `x↔p`-symmetric sibling, Berry & Keating's

    H_II = (x + ℓ_x²/x)(p + ℓ_p²/p),   x, p > 0,

restores the exchange symmetry `H_I` breaks. It belongs to the same general family
`H = U(x)p + ℓ_p²V(x)/p` (Sierra 2019 review, eq. 5.5) with `U = V = x + ℓ_x²/x`.

**It has no closed-form secular equation.** `H_I`'s Bessel-`K` secular equation
exists only because its associated 1+1D metric is *flat* (Rindler, scalar curvature
`R = 0`); `H_II`'s metric is *curved* (`R = −4ℓ_x²/(x(x²+ℓ_x²))`, eq. 6.7), so the
eigenproblem is not exactly solvable. No cached source gives an `H_II` secular
equation, and the change of variables that linearises `H_I`'s local term leaves
`H_II`'s nonlocal kernel non-exponential — it does **not** reduce to the Bessel
problem. Sierra himself works with the flat `H_I` for the solvable Dirac-ization
"because the flatness … makes the computations easier" (§VII). So the #23 → #31
*reference-first* template does not carry over: there is nothing to root-find.

**What is exactly computable is the semiclassical count** (#59,
`deformed_xp_symmetric`). The orbits `H_II = E` are closed loops around the fixed
point `(ℓ_x, ℓ_p)`, the classical floor is `H_II(ℓ_x, ℓ_p) = 4h` (twice `H_I`'s
`2h`, with `h = ℓ_xℓ_p`), and the enclosed phase-space area collapses — only the
product `h` enters, the same scaling symmetry as `H_I` — to a one-dimensional
integral:

    A(E) = 4h ∫₀^{arccosh B} √(B² − cosh²θ) dθ,   B = E / 4h,
    n_II(E) = A(E) / 2π  ~  (E/2π)(log(E/h) − 1) + …        (eq. 5.18)

**Verdict — the same average density as `H_I`, at the same scale.** `H_II`'s count
has the *same* leading two terms `(E/2π)(log(E/h) − 1)` as `H_I` (eq. 5.17) and as
the average zeros `N̄`, at the *same* `ℓ_xℓ_p = 2π` — restoring the `x↔p` symmetry
does **not** change the mean spectral density, and no rescaling is needed (the area
asymptotic is `log(E/h)`, matching Sierra eq. 5.18; verified numerically in the
tests). What the deformation *does* change is everything below the leading density:
the classical floor rises from `2h` to `4h`, and the subleading corrections differ
(eq. 5.17 vs 5.18). The `7/8` is absent (the semiclassical area pins the two leading
terms but not the `O(1)` constant — `n_II − N̄ → −7/8` here). Conclusion unchanged:
**average density yes, GUE fluctuations no** (Sierra: "no trace of the exact Riemann
zeros in the spectrum of the modified-`xp` models"). A full `H_II` quantum spectrum
would need a direct diagonalisation of the curved-metric operator — not pursued: the
semiclassical reading already settles the question, and the expected spectrum is the
same picket fence as `H_I` (#24).

## The numerics: CPU reference truth, GPU for scale (#23 → #31)

Per the house rule, the CPU secular reference is the truth and the GPU must match
it on the resolved low modes:

- **`deformed_xp.secular_spectrum`** (#23) — real roots of the Bessel-`K` secular
  equation (eq. 14), in fp64/`mpmath`. The reference spectrum.
- **`deformed_xp_gpu`** (#31) — assembles the operator as a dense Hermitian matrix
  (Galerkin) and diagonalises with `cupy.linalg.eigh` (cuSOLVER), reproducing the
  secular roots to floating-point precision for the resolved low modes. The GPU is
  for *scale* (larger matrices, more modes), not for changing the answer.

This is one of the few places in the repo where a single fp64 `eigh` is the right
tool: the deformed-`xp` low spectrum lives in an `O(1)`, well-conditioned regime,
unlike the flagship operator whose minimal eigenvalue drops below fp64 epsilon
almost immediately (contrast [`ccm-operator.md`](ccm-operator.md)).

## Reproduce

    uv run python scripts/run_deformed_xp.py            # CPU secular spectrum (#23)
    uv run python scripts/run_deformed_xp_gpu.py        # GPU eigh, asserts CPU match (#31)
    uv run python scripts/run_deformed_xp_stats.py      # universality: picket fence vs GUE (#24)
    uv run python scripts/run_deformed_xp_symmetric.py  # symmetric sibling: semiclassical count (#59)

The statistics sweep is marked `slow` (`uv run pytest -m slow` to include its
test); the spectra and GPU agreement run in the default suite.

## Sources

- **arXiv:1102.5356** — G. Sierra & J. Rodríguez-Laguna, *The H=xp model revisited
  and the Riemann zeros*, PRL **106**, 200201 (2011). The deformed operator `H_I`
  and its Bessel-`K` secular equation. (Local PDF in `_private/papers/`.)
- **arXiv:1601.01797** — G. Sierra, *The Riemann zeros as spectrum and the Riemann
  hypothesis*, Symmetry **11**(4), 494 (2019). §V: the general family
  `H = U(x)p + ℓ_p²V(x)/p` (eq. 5.5), its quantization (eqs. 5.6/5.8) and
  self-adjointness (eq. 5.10); the symmetric `H_II` (eq. 5.4), its counting function
  (eq. 5.18) and curved metric (§VI, eq. 6.7). The source for the `H_II` section
  above (#59). (Local PDF in `_private/papers/`.)
- Background on Berry–Keating `xp` and the average-zeros question: the predecessor
  archive [`predecessor/`](predecessor/) and [`project-framing.md`](project-framing.md).
