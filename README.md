# zeta-spectral-gpu

GPU/CUDA experiments on the **spectral approach to the Riemann zeta zeros**.

The spectral (Hilbert–Pólya) program asks for a self-adjoint operator whose
eigenvalues are the imaginary parts `τ` of the nontrivial zeros `½ + iτ`.
This repo is the *numerical* arm of that question: take operators and statistics
that are **structurally derived** and push them on a GPU to see what they do.

It is the GPU successor to the pure-Python research repo
`wedgetrigfunctions202601` (a private repo), which traced harmonic functions →
conical/Legendre functions → the hyperbolic (Selberg) spectrum → GUE spacing
statistics on CPU. Its conceptual notes are the prerequisite reading, distilled
public-safe into [`knowledge/predecessor/`](knowledge/predecessor/); this repo
reuses the conclusions and scales the compute.

## The one rule: forward, never inverse

There is a well-known trap in this area, and this repo is built around avoiding
it. The discriminating test for **any** idea here:

> **Does it consume the zeros as input?**
> If yes, it is curve-fitting (the Wu–Sprung trap) and teaches nothing —
> infinitely many operators reproduce any finite list of known zeros.
> The content of the problem is *producing* the zeros from something that
> **isn't them**.

- **Forward (valuable):** take a structurally derived operator (built from
  primes, geometry, a scaling action — *not* from the zeros) and ask "does its
  spectrum converge to the zeros?" The zeros are used **only to check output**.
- **Inverse (banned here):** take the zeros and fit an operator / potential to
  them. We do not do this, even though a GPU makes it easy.

Every experiment in this repo states, in its docstring, why it is on the
forward side of that line.

Three forward tracks. In every one, the zeros appear *only* as an output being
checked — never as input.

| Track | What | Status |
|---|---|---|
| **Warm-up statistics** | GPU spacing / pair-correlation / universality on large zero sets and candidate spectra. Extends `wedgetrigfunctions`' `gue_spacing.py` and scales it. Well-posed, falsifiable, embarrassingly parallel. | 🟢 landed: GUE nearest-neighbour spacing (#5), pair correlation vs the sine kernel (#6), spectral rigidity Σ²(L)/Δ₃(L) (#15), the De Bruijn–Newman forward heat flow (#20), and the unfolding-free spacing-ratio `r̃` (#35) |
| **xp-Hamiltonian track** *(forward stretch)* | Spectra of *principled* `xp`-type candidate operators: Sierra & Rodríguez-Laguna's geometrically deformed `xp` (#23/#24/#31) and the prime-driven massless-Dirac "Möbius-mirror" forward locator (#25). A contrast to the flagship — do geometry- and prime-built operators carry the mean density and/or the GUE fluctuations? | 🟢 deformed-`xp` CPU secular reference + GPU Galerkin eigensolve + universality read (picket-fence vs GUE); Dirac-mirror locator on CPU/GPU at scale (#38/#40), Dirichlet-`L` generalization (#42), RH-by-contradiction demo (#43). The genuinely-forward fluctuation question is the open spike #44. See `knowledge/deformed-xp.md`, `knowledge/dirac-mirror.md` |
| **Flagship** | Reimplement the **Connes–Consani–Moscovici finite-cutoff spectral-triple operators** (rank-one perturbation of a scaling operator; the matrix is the Weil explicit-formula quadratic form, whose prime content is the Euler/von-Mangoldt sum over primes `p ≤ x = λ²`) and study spectral convergence to the low zeros as the cutoff grows. Forward, prime-driven, on the live research edge. | 🟢 operator pinned (`knowledge/ccm-operator.md`, #3); multiprecision reference reproduces the source §6 table (`ccm.py`, #8); connes-cvs baseline oracle (#16); GPU fp64 assembly + λ-sweep convergence/conditioning study (`ccm_gpu.py`, #9); CPU-accel — factor-once + parity-reduced eigensolve, gmpy2 backend, ~6× (#17/#18); prime-cutoff rigidity trend toward GUE via the spacing-ratio `r̃` (`knowledge/ccm-universality.md`, #18) |

The flagship operator definition must come from the **primary source**
(Connes, Consani & Moscovici, *Zeta Spectral Triples*, arXiv:2511.22755, Nov
2025), not from any secondary note. Pinning it precisely was the first task of
that phase — now done; the verified, equation-by-equation spec lives in
`knowledge/ccm-operator.md`. Context for scale: that paper runs at
**~200-digit precision**
(matrix dimension 241, N=120) with per-zero errors down to ~1e-55 — which is
exactly why the GPU here is for assembly and sweeps, not the eigensolve.

## Stack

- **Harness:** Python (managed with [`uv`](https://docs.astral.sh/uv/)) + CuPy.
- **Eigensolvers:** `cupy.linalg.eigh` → cuSOLVER (we do not hand-roll linear
  algebra).
- **Hand-written kernels:** CUDA C via CuPy `RawModule`/`RawKernel` (NVRTC
  JIT) for the hot assembly / reduction paths — `src/zeta_spectral_gpu/kernels/`.
- **Reference precision math:** `mpmath` on CPU, for generating zeros and for
  any step where fp64 is not enough (see the precision note below).

### A precision caveat worth reading first

Consumer GPUs (this repo is developed on an RTX 3090) have **fp64 throughput
~1/64 of fp32**, and matching *low* zeta zeros to high accuracy can want
precision **beyond fp64**, which neither the GPU nor cuSOLVER provides natively.
So the GPU's real leverage here is **(a)** the embarrassingly-parallel
statistical experiments and **(b)** operator *assembly* (integer/fp32-friendly),
with the delicate high-precision eigenvalue convergence kept mixed (GPU
fp32/fp64 refinement + CPU/mpmath where precision bites). This is not "one giant
CUDA eigensolve"; it is a sweep harness plus assembly kernels.

The flagship's `ccm_gpu.py` (#9) makes that wall concrete: its fp64 matrix fill
reproduces the mpmath assembly to ~1e-12, but the operator's minimal eigenvalue
drops below fp64 epsilon almost immediately (≈1e-17 by cutoff `x = 5`), so a
double-precision eigensolve can recover the spectrum only at the smallest
cutoffs. `scripts/run_ccm_gpu.py` plots exactly where fp64 falls off, and keeps
the extended-precision eigensolve on the CPU.

## Running

Managed with [`uv`](https://docs.astral.sh/uv/). On a fresh machine install uv
(`winget install astral-sh.uv` on Windows; see the uv docs for macOS / Linux),
then from the repo root:

```powershell
uv sync                  # core (CPU) deps
uv sync --extra gpu      # add CuPy for the GPU paths
uv sync --extra oracle   # connes-cvs flagship cross-check oracle (#16, dev only)
uv sync --extra accel    # gmpy2 — faster multiprecision flagship eigensolve (#17)
uv run pytest            # CPU tests pass without a GPU; GPU tests self-skip
```

The `oracle` and `accel` extras are opt-in like `gpu`, so a plain `uv sync` (and
CI) stays lean. `oracle` is a dev/cross-check dependency only — never imported at
runtime (see `knowledge/connes-cvs-oracle.md`); `accel` is pure speed (`mpmath`
auto-detects `gmpy2` and the values are bit-identical with or without it).

uv reads `.python-version` (3.12) and provisions the interpreter itself — no
python.org / Microsoft Store Python needed. Run everything through `uv run`
(e.g. `uv run python scripts/run_spacing.py`).

The `gpu` extra pins `cupy-cuda12x[ctk]`; the **`[ctk]` is required**, not
optional — CuPy 14 JIT-compiles the CUDA kernels via NVRTC at runtime and needs
the CUDA toolkit headers, which `[ctk]` supplies as pip wheels (no system CUDA
Toolkit install). Verify it imports and sees the GPU rather than trusting the
pin:

```powershell
uv run python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

Developed and verified on an RTX 3090 (CUDA runtime 12.9 under a 13.2 driver). A
harmless `UserWarning: CUDA path could not be detected` appears on import because
the libraries come from pip wheels rather than a system toolkit — it does not
affect operation. See `CLAUDE.md` for more.

## Layout

```
src/zeta_spectral_gpu/   library code
  zeros.py               generate/cache Riemann zeros (mpmath) + unfolding
  spacing.py             CPU reference statistics — spacing, pair-correlation,
                         rigidity, the r̃ ratio (the GPU must match these)
  spacing_gpu.py         GPU statistics via CuPy + the RawModule kernel
  debruijn_newman.py     forward H_t-zero generator (mpmath) for the DBN flow (#20)
  deformed_xp.py         Sierra's deformed-xp operator, CPU secular reference (#23)
  deformed_xp_gpu.py     deformed-xp GPU dense eigensolve (Galerkin + eigh) (#31)
  dirac_mirror.py        prime-driven massless-Dirac Möbius-mirror locator (#25)
  dirac_mirror_gpu.py    GPU Möbius-mirror locator scan at scale (#25 Phase 2)
  dirichlet.py           Dirichlet characters / L-functions for the locator (#42)
  ccm.py                 Connes–Consani–Moscovici operator, mpmath reference (#8)
  ccm_gpu.py             fp64 fill + conditioning probe for the CCM operator (#9)
  oracle.py              loader for the connes-cvs baseline-oracle fixture (#16)
  plots.py               matplotlib figures for the warm-up + flagship statistics
  _cuda_dll.py           Windows cuSOLVER DLL-path shim for cupy.linalg.eigh
  kernels/spacing.cu       hand-written CUDA C — spacing reduction (first kernel)
  kernels/ccm_assembly.cu  hand-written CUDA C — Weil-matrix fill
  kernels/dirac_mirror.cu  hand-written CUDA C — Möbius mirror-locator sum
scripts/                 runnable entry points (uv run)
tests/                   invariants, incl. GPU-vs-CPU agreement
knowledge/               conceptual notes (why the math looks the way it does)
data/                    generated/cached zeros (gitignored)
```

## License

[MIT](LICENSE) © 2026 Skyline Trail Computing LLC.
