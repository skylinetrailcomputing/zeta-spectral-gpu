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
uv-managed. On a fresh machine: install uv (`winget install astral-sh.uv` on
Windows), then from the repo root `uv sync` (CPU deps) and `uv sync --extra gpu`
(adds CuPy). uv reads `.python-version` (3.12) and provisions the interpreter
itself — no python.org / Microsoft Store Python needed. Run things via `uv run`
(e.g. `uv run pytest`, `uv run python scripts/run_spacing.py`).

The `gpu` extra is `cupy-cuda12x[ctk]`: the **`[ctk]` is required**, not optional —
CuPy 14 JIT-compiles kernels via NVRTC at runtime and needs CUDA toolkit headers,
which `[ctk]` supplies as pip wheels (no system CUDA Toolkit install). Verified on
an RTX 3090 (CUDA runtime 12.9 under a 13.2 driver). A harmless
`UserWarning: CUDA path could not be detected` appears on import because the libs
come from pip wheels rather than a system toolkit — it does not affect operation.

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
