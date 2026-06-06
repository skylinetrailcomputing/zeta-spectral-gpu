# zeta-spectral-gpu

GPU/CUDA experiments on the **spectral approach to the Riemann zeta zeros**.

The spectral (Hilbert–Pólya) program asks for a self-adjoint operator whose
eigenvalues are the imaginary parts `τ` of the nontrivial zeros `½ + iτ`.
This repo is the *numerical* arm of that question: take operators and statistics
that are **structurally derived** and push them on a GPU to see what they do.

It is the GPU successor to the pure-Python research repo
[`wedgetrigfunctions202601`](https://github.com/bradleypmartin/wedgetrigfunctions202601),
which traced harmonic functions → conical/Legendre functions → the hyperbolic
(Selberg) spectrum → GUE spacing statistics on CPU. The conceptual notes there
are the prerequisite reading; this repo reuses the conclusions and scales the
compute.

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

## Roadmap

| Phase | What | Status |
|---|---|---|
| **Warm-up** | Spacing / pair-correlation / universality statistics on large zero sets and candidate spectra, on the GPU. Extends `wedgetrigfunctions`' `gue_spacing.py` and scales it. Well-posed, falsifiable, embarrassingly parallel. | 🟡 scaffolding |
| **Flagship** | Reimplement the **Connes–Consani–Moscovici finite-cutoff spectral-triple operators** (rank-one perturbation of a scaling operator; the matrix is the Weil explicit-formula quadratic form, whose prime content is the Euler/von-Mangoldt sum over primes `p ≤ x = λ²`) and study spectral convergence to the low zeros as the cutoff grows. Forward, prime-driven, on the live research edge. | ⚪ paper-read first |

The flagship operator definition must come from the **primary source**
(Connes, Consani & Moscovici, *Zeta Spectral Triples*, arXiv:2511.22755, Nov
2025), not from any secondary note. Pinning it precisely is the first task of
that phase. Context for scale: that paper runs at **~200-digit precision**
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

## Running

> **Environment is not yet stood up on the dev machine.** Build it first:
>
> ```powershell
> # install a real Python 3.11+ and uv, then:
> uv sync                  # core (CPU) deps
> uv sync --extra gpu      # add the CuPy wheel matching the installed CUDA
> uv run python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
> ```
>
> The exact CuPy wheel (`cupy-cuda12x` vs `cupy-cuda13x`) must match the
> installed CUDA runtime — **verify it imports and sees the GPU** rather than
> trusting the pin. See `CLAUDE.md`.

## Layout

```
src/zeta_spectral_gpu/   library code
  zeros.py               generate/cache Riemann zeros (mpmath) + unfolding
  spacing.py             CPU reference statistics (the GPU must match these)
  spacing_gpu.py         GPU statistics via CuPy + the RawModule kernel
  kernels/spacing.cu     hand-written CUDA C (first kernel target)
scripts/                 runnable entry points (uv run)
tests/                   invariants, incl. GPU-vs-CPU agreement
knowledge/               conceptual notes (why the math looks the way it does)
data/                    generated/cached zeros (gitignored)
```

## License

TODO — decide before the first tag (MIT or Apache-2.0 are the likely picks).
Until then, all rights reserved by Skyline Trail Computing LLC.
