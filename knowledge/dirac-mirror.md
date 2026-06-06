# Sierra's prime-driven Möbius-mirror model — a forward locator (#25)

*Why* Sierra's massless-Dirac "moving-mirror" model belongs here, what its forward
**locator** does, the **circularity** that reshaped the issue (no single-operator
spectrum), what landed across four phases, and the one genuinely-forward
fluctuation question that remains open (#44). Read
[`project-framing.md`](project-framing.md) first; this is the prime-driven half of
the **xp-Hamiltonian track** and the more ambitious forward contrast to the
flagship — primes enter as periodic-orbit periods `log p`.

## Why this is forward, not inverse

The reflection coefficients are the **Möbius function** `μ(n)/√n` (built from the
primes), and `1/ζ(s) = Σ_{n≥1} μ(n)/n^s`. The truncated partial sum of that series
is the locator; `|·|` grows at the zeros of `ζ` because `1/ζ` blows up there. The
zeros are an **output** we then check against an independently-computed zero list.
No zeros are consumed → forward.

## The object: the forward locator

A massless Dirac fermion in the right Rindler wedge (radial `ρ ≥ ℓ₁ = 1`) is free
except for delta-function **"moving mirrors"** at radial positions `ℓ_n`, each a
unitary 2×2 transfer matrix `T_n` with reflection coefficient `ϱ_n` (eq. 10.20).
Amplitudes iterate from the boundary vector `|A₁(ϑ)⟩ = (1, e^{iϑ})` (eq. 10.19/22).
The **prime-driven** choice (eq. 11.4 / 13.5) puts a mirror at every square-free
integer with strength `μ(n)/√n`. The decisive forward object is the truncated
partial sum `M'_z(n)` (eq. 12.20): scan it over an energy grid `E` and `|M'_z(n)|`
**peaks at the ζ ordinates**. That peak-finder is the *locator*. It generalises
verbatim to any Dirichlet `L`-function by `μ(n)/√n → χ(n) μ(n)/√n` (since
`1/L(s,χ) = Σ χ(n) μ(n)/n^s`), locating the `L(s,χ)` zeros.

## The circularity that reshaped the issue (read this)

The originally-planned "GUE spacing readout of the spectrum" was **dropped as
circular**, and this is the key conceptual result of #25:

> The model has **no single-operator spectrum**. Each zero is an eigenvalue of a
> *different* self-adjoint extension `H_ϑ` (Sierra §XV, the "local" Pólya–Hilbert
> picture; cf. Fig. 16, the Connes missing-lines view). So the locator's peaks
> *are* the known zeros — feeding them to the spacing / `Σ²` / `Δ₃` harness just
> re-measures the zeros' own statistics that #5/#6/#15 already established.

Hence the locator is a beautiful forward *detector* of the zeros, but its peak
*spacings* carry no new content. The genuinely-forward fluctuation question lives
one level down, in the finite-`ε` object (below, #44).

## What landed (four phases)

- **Phase 1 — CPU forward locator (#38).** `dirac_mirror.mobius_partial_sum` and
  the peak-finder; numpy reference. The truth the GPU must match.
- **Phase 2 — GPU scan at scale (#40).** `dirac_mirror_gpu` + the hand-written
  `kernels/dirac_mirror.cu`: one thread per `E` point with an inner loop over `k`,
  doing the `O(len(E)·n)` work in `O(len(E)+n)` memory (the numpy reference
  materializes the full `len(E)×n` phase matrix — the memory wall). fp64
  throughout: the locator lives in an `O(1)`, cancellation-free regime, so double
  precision suffices and reproduces the CPU reference to floating point.
- **Phase 3 — Dirichlet-`L` generalization (#42).** `dirichlet.py` adds Dirichlet
  characters and `1/L(s,χ)`; real characters use the fast real kernel, a genuinely
  complex character uses the `weighted_locator` kernel and shows the
  asymmetric-in-`E` zeros of a complex `L`-function.
- **Phase 4 — RH-by-contradiction demo (#43; Sierra §XII C).** A zero *on* the
  line makes the truncated Möbius sum grow only like `log n` (eq. 12.30), so the
  bound-state norm is finite once `ϑ` is tuned (eq. 12.34). A hypothetical zero
  *off* the line (`σ_c > ½`) would make `|M_z(n)|` grow polynomially `~ n^{σ_c−½}`
  (eq. 12.35), and the norm then diverges for **every** `ϑ` — which a self-adjoint
  `H_ϑ` cannot do. The off-line zero is a **planted counterfactual** (no true zero
  is consumed → still forward); the script exhibits both growth laws numerically.

## The open forward question (#44)

The finite-`ε` object — the full transfer-matrix product `T_n···T₂` (eq. 10.20 /
12.6), *not* the linearized `ε→0` locator — has a resonance / density-of-states
structure built from the **periodic orbits** (primes `p` as orbits of period
`log p`, amplitudes `μ(n)/√n`; Berry's primes-as-orbits, eq. 11.5). The forward
statistic is the semiclassical two-point correlation / spectral form factor of
that density of states, *compared* to GUE — no zeros consumed. Sharpest prediction:
truncating to primes/orbits up to a cutoff `P` should make RMT-tracking extend to a
range set by the longest orbit `log P` — a saturation scale mirroring #15's Berry
scale `L* ≈ ln(T/2π)/π`. This is the fluctuation content the locator could not
deliver; spun out as the open spike **#44** (a real lift: needs a resonance /
complex-pole search, not `eigh`).

## Reproduce

    uv run python scripts/run_dirac_mirror.py        # CPU forward locator (#38)
    uv run python scripts/run_dirac_mirror_gpu.py    # GPU scan, asserts CPU match (#40)
    uv run python scripts/run_dirichlet_mirror.py    # Dirichlet-L locator (#42)
    uv run python scripts/run_rh_contradiction.py    # on-line vs off-line growth (#43)

## Sources

- **arXiv:1404.4252** — G. Sierra, *The Riemann zeros as energy levels of a Dirac
  fermion in a potential built from the prime numbers in Rindler spacetime*,
  J. Phys. A **47**, 325204 (2014). The mirror model, locator, and §XII C RH
  argument. (Local PDF in `_private/papers/`.)
- **arXiv:1601.01797** — Sierra, review of the `H=xp`/Riemann-zeros program,
  §X–XV (the local Pólya–Hilbert / self-adjoint-extension picture and Fig. 16).
- **arXiv:0712.0705** — the 2008 precursor.
