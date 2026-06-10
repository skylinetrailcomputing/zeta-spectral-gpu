# Small-gap / Lehmer-pair census at height (#86)

The first *science consumer* of the #55 GPU Riemann–Siegel evaluator
(`riemann_siegel_gpu`): scan `Z(t)` over long height windows, harvest the
critical-line zeros from sign changes (primes/analysis in, zeros out —
forward), and read the consecutive-gap list two ways:

1. the **small-gap tail** of the normalized spacing distribution against the
   GUE level-repulsion law `p(s) → (π²/3)s²` as `s → 0`;
2. a **Lehmer-pair census** under the Csordas–Smith–Varga (CSV 1994)
   criterion, each qualifying pair yielding a forward-computed lower bound on
   the De Bruijn–Newman constant `Λ` (`= 0` under RH by Rodgers–Tao; see
   [`debruijn-newman-flow.md`](debruijn-newman-flow.md)).

Code: `lehmer_census.py` (evaluator-injectable scan + census; the GPU path is
just `hardy_z_gpu` plugged into the same machinery), script
`scripts/run_lehmer_census.py`, figure `plots.lehmer_census_figure`.

## The CSV criterion, and a normalization pin worth recording

For consecutive simple zeros `γ₋ < γ₊` put `Δ = γ₊ − γ₋` and

```
g = Σ_{γ ≠ γ₋,γ₊} 1/(γ−γ₋)² + 1/(γ−γ₊)²     (all other zeros of Ξ, ±γ_j)
```

The pair is a **Lehmer pair** iff `Δ²g < 4/5`, and then (Stopple 2017, eqs.
2–5, in zeta-ordinate coordinates)

```
λ(Δ, g) = ((1 − 5Δ²g/4)^{4/5} − 1) / (8g)  ≤  Λ.
```

**The trap:** that formula is in the flow normalization whose time-zero zeros
sit at the ordinates `γ` themselves. Every published `Λ` table (de Bruijn's
`≤ 1/2`, CSV's `−4.379e−6`, COSV's `−5.895e−9`, Polymath15's `≤ 0.22` — and
this repo's #20 `H_t`) uses the classical flow with zeros at `2γ`, related by
`H_t(z) ∝ Ξ_{t/4}(z/2)`, so **published λ = 4 × the γ-coordinate formula**
(`Δ²g` itself is dimensionless and identical in both). The factor is pinned
empirically, not just derived: for a high-quality pair
`λ ≈ −Δ²/2 + O(Δ⁴g)` (classical units, nearly `g`-independent), and the COSV
1993 pair — `Δ ≈ 1.0857e−4` at `t ≈ 3.8886e8`, reconstructed from Stopple's
§6 data — gives `−Δ²/2 = −5.894e−9` against the published `−5.895e−9`.
`csv_lambda` returns classical units. (Stopple's own exposition quotes the
classical numbers next to the γ-coordinate formula without flagging the
factor; transcribe with care.)

The censused `g` is completed outside the scanned window (and on the whole
mirror axis `γ < 0`) by closed-form integrals of the smooth density
`ρ(x) = log(x/2π)/2π` — a per-mille correction at the window sizes used, but
it makes the per-pair `λ` a *measured census readout*, not a certified bound
(CSV-era certificates bracket the tail rigorously).

## Precision budget (written down before scanning, per the house rule)

fp64 `Z` carries phase-rounding error `ε(t) ≈ (eps/2)·t·log(t/2π)` (the #55
ceiling; the script re-measures against `mpmath.siegelz` — the model is
order-of-magnitude, measured values run within ~5× of it). A close pair with
normalized gap `s` leaves an interior extremum `|Z_ext| ~ (π²/8)·A·s²` (`A` =
local RMS amplitude), so the resolvable-gap floor is

```
s_min(t) = sqrt(8·safety·ε / (π²·A))      (gap_resolution_floor)
```

| height | ε model | s_min (A=2.5) |
|---|---|---|
| 1e5 | ~1e-10 | ~2e-5 |
| 1e6 | ~1.3e-9 | ~6e-5 |
| 1e7 | ~1.6e-8 | ~2e-4 |
| 1e8 | ~1.8e-7 | ~7e-4 |

The GUE cube law makes gaps below these floors one-in-a-million events, so
**fp64 is nowhere near binding for a feasible census at `t ≤ 1e7`** — the
binding constraint is scan volume. But *record-quality* pairs live exactly
under the floor: the COSV pair (`s ≈ 3.1e−4` at `t ≈ 3.9e8`) is unresolvable
in fp64 — its interior extremum (~3e−7) sits under the phase noise (~8e−7).
The pipeline's answer is tiered: coarse sign scan → `|Z|`-dip rescan at a
finer grid (recovers pairs to `s ~ fine_fraction`) → `near_misses` +
`below_floor` flags handed to `mpmath` (`polish_pair` / `resolve_near_miss`).

## Results (RTX 3090, fp64 scan + mpmath shortlist polish)

Three windows, ~255k zeros, ~90 s of GPU scan total
(`run_lehmer_census.py --cosv`, 2026-06-09):

| window | zeros | count deficit | Lehmer pairs | rate | best λ ≤ Λ |
|---|---|---|---|---|---|
| [1e5, 1.6e5] | 94,809 | +0.2 | 5,820 | 6.14% | −2.5e−4 |
| [1e6, 1.06e6] | 114,658 | +0.9 | 7,385 | 6.44% | −1.9e−4 |
| [1e7, 1.002e7] | 45,458 | +0.5 | 2,927 | 6.44% | −4.8e−5 |

- **Forward reproduction of Stopple's census.** The 1e6 window is exactly his
  §7 range: he reports **7398 / 114,661** Lehmer pairs; the forward scan finds
  **7385 / 114,657** (99.8% — the residual is edge bookkeeping and his
  Mathematica zero list vs ours). No zero data was input anywhere.
- **Forward reproduction of the COSV 1993 record bound.** At
  `t ≈ 3.8886e8` the fp64 scan correctly *fails closed*: count deficit ~1.4
  with one `near_miss` dip flagged (the pair's extremum ~3e−7 is under the
  ~8e−7 phase noise; floor `s_min ≈ 1.1e−3` vs the pair's `s ≈ 3.1e−4`).
  `resolve_near_miss` (mpmath, 30 dps, ~12 s) then resolves
  `(388858886.0022851, 388858886.0023937)`, `Δ = 1.0854e−4`, and the census
  formula returns `λ = −5.890e−9` against the published `−5.895e−9` (0.1%,
  the residual from the density-only `g` used at that isolated height).
- **Small-gap tail vs GUE, with height.** The empirical cumulative tail runs
  *below* the GUE laws at low height and converges upward: at `s < 0.5` the
  deficit against the Wigner-surmise CDF is 11% → 6.9% → 6.3% across
  1e5 → 1e6 → 1e7 (the deepest tail `s < 0.1` matches the cube law within
  Poisson noise at all heights). Same moral as the #84 pair-correlation read:
  low height shows the arithmetic (lower-order) deviation, and the census
  measures its decay — the "universality with height" readout the issue
  asked for.
- **The classical Lehmer pair** (`γ_6709/6710` at `t ≈ 7005`) comes out of a
  blind window scan as the top-quality row: `(7005.062866, 7005.100565)`,
  `s = 0.0421`, `Δ²g = 0.0069` (far under the 4/5 bar), `λ = −7.1e−4`.
- **fp64 vs mpmath polish:** at `t ≤ 1e7` the shortlisted pairs' zeros move
  by ≤ 3e−8 under a 30-dps re-resolve and `λ` is stable to 5–6 digits — fp64
  is comfortably inside the written budget; the budget binds only for
  COSV-class record pairs (above) or heights past ~1e8.
- Scan health: GPU-vs-CPU `max|ΔZ|` 2.4e−10 / 3.1e−9 / 3.9e−8 per window
  (tracking the phase-error model); dip-rescue recovered 4+4 sub-grid pairs
  at 1e5/1e6; every window's count deficit is within `S(t)` fluctuation.

## Forward shape

The zeros consumed by every statistic are produced by the Riemann–Siegel scan
itself — no zero data enters as input. The classical Lehmer pair
(`γ_6709/6710`), `mp.zetazero`, and the published COSV/Stopple numbers appear
only as after-the-fact checks (tests + script validation blocks). A censused
Lehmer pair is therefore a *forward-computed certificate-shaped statement*
about `Λ` — the same constant the #20 heat-flow spike approached from the
flow side, now reached from the zero-gap side through nothing but the
evaluator.

## Sources

- Csordas, Smith & Varga, Constr. Approx. **10** (1994) — the criterion.
- Csordas, Odlyzko, Smith & Varga, ETNA **1** (1993) — the COSV pair.
- Stopple, Exp. Math. **26** (2017), arXiv:1508.05870 — the restatement
  transcribed here; §7 counts; local PDF in `_private/papers/`.
- See [`bibliography.md`](bibliography.md) (#86 section).
