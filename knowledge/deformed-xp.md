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
`x↔p`-symmetric sibling `H = (x + ℓ_x²/x)(p + ℓ_p²/p)` — same conclusion below.

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

    uv run python scripts/run_deformed_xp.py         # CPU secular spectrum (#23)
    uv run python scripts/run_deformed_xp_gpu.py     # GPU eigh, asserts CPU match (#31)
    uv run python scripts/run_deformed_xp_stats.py   # universality: picket fence vs GUE (#24)

The statistics sweep is marked `slow` (`uv run pytest -m slow` to include its
test); the spectra and GPU agreement run in the default suite.

## Sources

- **arXiv:1102.5356** — G. Sierra & J. Rodríguez-Laguna, *The H=xp model revisited
  and the Riemann zeros*, PRL **106**, 200201 (2011). The deformed operator and its
  Bessel-`K` secular equation. (Local PDF in `_private/papers/`.)
- Background on Berry–Keating `xp` and the average-zeros question: the predecessor
  repo `wedgetrigfunctions202601` and [`project-framing.md`](project-framing.md).
