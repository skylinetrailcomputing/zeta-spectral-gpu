# The connes-cvs baseline oracle (#16)

> The first concrete coding step of the flagship track: a **validation oracle**,
> stood up *before* we write any of our own assembly. `connes-cvs` (Akiva Groskin,
> 2026; PyPI, MIT) is the first public implementation of the Connes–van Suijlekom
> Galerkin matrix `Q(c)`. We reproduce its documented `c=13` cell and freeze the
> result as a checked-in fixture for the CPU reference (#8) and GPU assembly (#9)
> to diff against. Pairs with the operator spec in [`ccm-operator.md`](ccm-operator.md).
>
> **Forward, and used as an oracle only.** `connes-cvs` is itself forward — it
> builds `Q(c)` from the prime / von-Mangoldt sum and extracts zeros as *output*.
> Here it is a **dev / cross-check dependency only** (the opt-in `oracle` extra),
> never a runtime import. We compare our numbers to it; we never fit to it. See
> [`project-framing.md`](project-framing.md).

## The oracle cell

| | |
|---|---|
| Cell | `c=13, N=80, T=400, dps=80` (the documented A/B cell) |
| `lambda_min` | `2.5282661401965756026…e-59` (full value in the fixture) |
| `gamma_1` detected | `14.13472514173469379045725198356…` — the true `t_1` to ~54 digits |
| `|gamma_1 - t_1|` | `1.7711953691758599…e-55` |
| Package | `connes-cvs==0.2.2` + `python-flint==0.8.0` (Arb digamma) |

Reproduce / refresh:

```bash
uv sync --extra oracle
uv run python scripts/run_connes_cvs_oracle.py            # verify against target
uv run python scripts/run_connes_cvs_oracle.py --write    # rewrite the fixture
uv run pytest -m slow                                      # rebuild + assert match
```

The fixture lives at `tests/fixtures/connes_cvs_c13_oracle.json` and is loaded by
`zeta_spectral_gpu.oracle` (loading needs only `mpmath`, not `connes-cvs`).

## Two numbers, kept distinct

It is easy to conflate the ground-state eigenvalue with the zero error; they are
different quantities and differ by orders of magnitude:

- **`lambda_min(c)`** — the smallest eigenvalue of `Q(c)`, the *Weil-positivity
  proxy*. At `c=13`, `≈ 2.53e-59`. This is the oracle target.
- **`|gamma_1 - t_1|`** — the *first-zero error*, how far the eigenvector's
  recovered zero sits from the true ordinate. At this cell `≈ 1.77e-55`.

`lambda_min → 0` super-exponentially in `c`; the zero error shrinks too but lives
several orders of magnitude higher. fp64 (`~1e-16`) cannot represent either, so
this is a CPU/`mpmath` computation — there is nothing to put on the GPU at #16.

## The normalization-convention factor (~1.2–1.3)

The published papers tabulate the **zero error**, not `lambda_min`. So absolute
`lambda_min` is only meaningful relative to a convention. The cross-check that
fixes the convention is the first-zero error at `c=13`:

| Source | `|gamma_1|` error at `c=13` | vs connes-cvs |
|---|---|---|
| connes-cvs (reproduce-paper: N=100, T=800, dps=150) | `2.005e-55` | 1.00 |
| CCM 2025 §6 (N=120, 200 dps; arXiv:2511.22755) | `2.44e-55` | ×1.22 |
| Connes 2026 (first-50 data; arXiv:2602.04022) | `2.6e-55` | ×1.30 |

The ~1.2–1.3 spread is attributed by Groskin to differing `N / T / precision` and
normalization conventions, **not** a correctness gap (connes-cvs is the third
independent measurement at `c=13`). Our A/B cell (N=80, T=400, dps=80) gives a
first-zero error of `1.77e-55` — a *different* cell than the reproduce-paper one
above, so its absolute number differs again; the table is the convention anchor,
recorded so that when #8/#9 produce their own `lambda_min` we know which factor to
expect before trusting the absolute value.

## How connes-cvs builds Q(c) (for #8 cross-reference)

From its "How it works" notes (matching the CvS/CCM construction):

```
Q(c) = D_inf + D_pole + D_prime
```

- `D_inf` — archimedean Mellin multiplier, `h_+(τ) = Re ψ(1/4 + iτ/2) − log π`
  (digamma `ψ`). **The cost driver:** ~2.2M high-precision complex-digamma evals,
  ~80% of wall time; the eigensolve is only ~17%. (This profiling reality
  motivated the CPU-accel work #17/#18, which landed a factor-once, parity-reduced
  multiprecision eigensolve plus the gmpy2 backend — ~6× on the flagship overall.)
- `D_pole` — rank-one correction from the pole of `ζ` at `s = 1`.
- `D_prime` — finite von-Mangoldt sum over primes `p ≤ c`.

The Galerkin basis is **trigonometric** (`sin(2π n y / L)`, `L = log c`; CCM 2025
Lemma 5.1), *not* prolate-spheroidal — connes-cvs's README carries an explicit
correction on this point. This matches §3 of our [`ccm-operator.md`](ccm-operator.md).

## Sources

- **connes-cvs** — A. Groskin, 2026. PyPI; MIT. Repo
  `github.com/akivag613/connes-cvs-`; Zenodo DOI 10.5281/zenodo.19546514. API:
  `build_galerkin_matrix(c, N, T, dps)`, `compute_ground_state(Q) -> (lambda_min,
  eigvec)`, `extract_zeros(eigvec, L, n_zeros, dps)`.
- **arXiv:2511.23257** — Connes & van Suijlekom, *Quadratic Forms, Real Zeros and
  Echoes of the Spectral Action*. The `Q(c)` (Prop. 4.1) that connes-cvs implements.
- **arXiv:2511.22755** — Connes, Consani, Moscovici, *Zeta Spectral Triples*. §6
  numerical tables (the zero-error column the factor above is measured against).
- **arXiv:2602.04022** — Connes, *The Riemann Hypothesis: Past, Present and a
  Letter Through Time*. First-50-zeros data at `c=13`.
