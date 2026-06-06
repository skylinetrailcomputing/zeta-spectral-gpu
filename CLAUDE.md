# zeta-spectral-gpu — project notes (for Claude & contributors)

## What this is
GPU/CUDA experiments on the spectral approach to the Riemann zeta zeros. The
GPU successor to the CPU repo `wedgetrigfunctions202601`. Read the `README.md`
and `knowledge/` before changing anything.

## The hard project rule: forward, not inverse
Every experiment must be **forward** — a structurally derived operator/statistic
whose output is *compared* to the zeros — never **inverse** (fitting an operator
or potential to the known zeros). The test: *does it consume the zeros as input?*
If yes, it does not belong here. This is not a style preference; it is the whole
point of the project (see `README.md` and `knowledge/project-framing.md`).

## Stack & conventions
- Python managed with `uv`; CuPy for GPU. Eigensolves via `cupy.linalg.eigh`
  (cuSOLVER) — do not hand-roll linear algebra.
- Hand-written CUDA C kernels live in `src/zeta_spectral_gpu/kernels/` and are
  loaded with CuPy `RawModule` (NVRTC JIT). Kernels are for hot assembly /
  reduction paths only — the deliberate "learn CUDA" surface.
- Style: line length **88**, `ruff format`, comments only when the *why* isn't
  obvious. Tests for new logic (`pytest`). Conventional Commits.
- **CPU reference first:** every GPU statistic has a CPU reference (`spacing.py`)
  and a test asserting GPU-vs-CPU agreement on small N. The GPU is for *scale*,
  not for changing answers.

## Precision reality (don't forget)
Dev GPU is an RTX 3090: fp64 ≈ 1/64 of fp32. Matching low zeta zeros can need
precision beyond fp64. GPU leverage = parallel statistics + operator assembly;
keep delicate high-precision eigenvalue work mixed (CPU/`mpmath` where needed).
Don't frame a task as "one giant fp64 eigensolve" without checking the
precision budget first.

## Environment
Build with `uv sync` then `uv sync --extra gpu`. The CuPy wheel must match the
installed CUDA runtime — the dev box reports a **CUDA 13.2 driver** (backward
compatible with the 12.x runtime, so `cupy-cuda12x` is expected) but **verify**
`import cupy` sees the GPU; don't trust the pin. As of scaffolding there is no
Python/uv on the dev machine yet — standing up the runtime is the gating task
before any kernel runs.

## Branch protection
Public repo under `skylinetrailcomputing/`. Standard Skyline OSS protections
apply (PR-before-merge ruleset, no force-push to default, secret scanning +
push protection). Branch + PR for every change; the owner self-merges after CI.

## A note on local context
This repo carries a git-ignored `CLAUDE.local.md` with maintainer-only context
(workspace imports, author identity, the multi-agent autonomy policy). It is not
required to work on the public code, and nothing in it changes the public
behavior described above — it only wires this repo into its maintainer's private
orchestration layer.
