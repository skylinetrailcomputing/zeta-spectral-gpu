# The quantum-chaos map — keying the QC toolkit to this repo's tracks (#88)

The "quantum chaos camp" — Berry, Keating, Bogomolny, Sierra, and the
semiclassical school — is where most of this repo's *falsifiable predictions*
come from. Its central dictionary (the Riemann zeros as the spectrum of an
unknown chaotic Hamiltonian whose periodic orbits are the primes) underlies the
warm-up statistics, the xp track, and the flagship alike, but until now it was
scattered across five notes. This is the one-page map: each QC concept, its
zeta-side counterpart, and where (or whether) the repo exercises it.

Forward-rule status: the dictionary itself consumes no zeros — it is the
*source of predictions* the zeros are checked against. Each linked experiment
states its own forward case.

## The dictionary

| quantum chaos | zeta side | repo artifact |
|---|---|---|
| chaotic Hamiltonian `H` | the conjectured Hilbert–Pólya operator | the flagship CCM operator ([`ccm-operator.md`](ccm-operator.md)) |
| Gutzwiller trace formula | the Weil / Riemann explicit formula | the Weil quadratic form the CCM matrix *is*; the Dirac-mirror orbit reading ([`dirac-mirror.md`](dirac-mirror.md)) |
| periodic orbit, period `T_γ` | prime power `p^m`, period `log p^m` | primes-as-orbits (Sierra eq. 11.5; [`dirac-mirror.md`](dirac-mirror.md)) |
| orbit stability amplitude | `Λ(n)/√n` | the von Mangoldt weights in every prime-built operator here |
| mean density (Weyl term) | Riemann–von Mangoldt `N(T)` | unfolding in `zeros.py`; the deformed-`xp` mean-density match ([`deformed-xp.md`](deformed-xp.md)) |
| no time-reversal symmetry ⇒ GUE (not GOE) | Montgomery–Odlyzko | spacing #5, pair correlation #6, `r̃` #35 |
| BGS universality conjecture | "the zeros are GUE because the dynamics is chaotic" | the warm-up baseline ([`predecessor/`](predecessor/)) |
| arithmetic-chaos exception (Hecke symmetries ⇒ Poisson) | modular Maass spectrum | the predecessor headline ([`predecessor/computational-arc.md`](predecessor/computational-arc.md)) |
| diagonal approximation (Berry 1985) | Montgomery's theorem range; `Σ²` saturation `L*` | rigidity #15 |
| off-diagonal orbit pairs | Hardy–Littlewood prime pairs ⇒ lower-order `R₂` | **#84 (filed)** — Bogomolny–Keating / Conrey–Snaith |
| form factor `K(τ)` arithmetic marks | departures at `τ = log(p^m)/2π` | **#84**; the comparison target for the #44 spike |
| Riemann–Siegel-lookalike resummation ("a rule for quantizing chaos") | the Riemann–Siegel formula itself | the #55 evaluator ([`riemann-siegel.md`](riemann-siegel.md)) |
| `H = xp` and its regularizations | Berry–Keating / Connes semiclassics | the deformed-`xp` track #23/#24/#31/#59 |
| rank-one point scatterer (Šeba class), intermediate statistics | — (an operator-side tool) | **#87 (filed)** — the CCM pole-locked tail |
| negative controls (systems where the conjecture fails) | functional equation without Euler product | **#85 (filed)** — Davenport–Heilbronn |
| spectral determinants / characteristic polynomials | Keating–Snaith moments | **out of charter** — value distribution, not a spectrum ([`frontier-survey-2026.md`](frontier-survey-2026.md)) |

## The rows worth expanding

**The trace-formula dictionary is exact, not an analogy.** The explicit formula
*is* a trace formula: a sum over the spectrum (zeros) equals a smooth (Weyl)
term plus a sum over primes with weights `Λ(n)/√n` and "periods" `log n` —
formally identical to Gutzwiller's spectrum = smooth + orbit sum. The unknown
is only which Hamiltonian. That identification is why the flagship matrix (the
Weil form, prime side in, spectrum out) and the Dirac-mirror locator are the
same bet made with different machinery, and why `2π/log x` keeps appearing as
a resolution scale (the longest-orbit cutoff): the #15 saturation scale, the
`k*(x)` pole-density argument in [`ccm-universality.md`](ccm-universality.md),
and the #44 prediction are all the same semiclassical statement.

**What the camp adds beyond universal RMT.** Universality (GUE spacing, sine
kernel, `Δ₃`) is the *generic* part — any chaotic system without time-reversal
matches it, so it cannot identify ζ. The QC program's sharper content is the
**system-specific corrections** built from the actual orbits, i.e. the primes:
Berry's `Σ²` beyond saturation, the Bogomolny–Keating lower-order pair
correlation, the form factor's arithmetic marks. That is the #84 ladder: the
repo has climbed the universal rungs; the arithmetic rungs are next. The same
logic at the family level is Katz–Sarnak
([`katz-sarnak-families.md`](katz-sarnak-families.md)) — symmetry type is the
universal part, finite-conductor terms the arithmetic part.

**Intermediate statistics is the camp's tool for *this repo's own operator*.**
Most of the dictionary points at the zeros; the Šeba / Bogomolny–Gerland–
Schmit intermediate-statistics theory instead describes **rank-one
perturbations** — the CCM operator's literal structure. It predicts, from the
coupling-vs-pole-spacing ratio alone, where a spectrum sits between picket
(weak coupling, roots pinned to poles) and the semi-Poisson class (strong
coupling). Applied to the prime-built CCM couplings that is a parameter-free
theory of the pole-locked tail and the `k*(x)` crossover — the #87 spike.

**What deliberately stays out.** Keating–Snaith / CFKRS moments and value
distribution (not spectral; see the frontier survey's verdict); the
Bender–Brody–Müller operator (non-self-adjoint; Sierra's review §IX); quantum
graphs (Kottos–Smilansky) — a lovely exactly-solvable QC laboratory, but here
at most a *pipeline-calibration* idea (a tunable GOE/GUE system with an exact
trace formula to validate the statistics code against known truth), not a
zeta experiment; noted, not filed.

## Sources

- M.V. Berry, *Semiclassical theory of spectral rigidity*, Proc. R. Soc. A
  **400** (1985) 229 — diagonal approximation, saturation.
- M.V. Berry & J.P. Keating, *The Riemann zeros and eigenvalue asymptotics*,
  SIAM Review **41** (1999) 236 — the canonical statement of the dictionary
  and of `H = xp`.
- M.V. Berry & J.P. Keating, *A rule for quantizing chaos?*, J. Phys. A **23**
  (1990) 4839 — the Riemann–Siegel-lookalike resummation.
- E. Bogomolny & J.P. Keating, *Random matrix theory and the Riemann zeros*
  I & II, Nonlinearity **8** (1995) 1115; **9** (1996) 911 — off-diagonal /
  Hardy–Littlewood lower-order terms (#84).
- P. Šeba, PRL **64** (1990) 1855; E. Bogomolny, U. Gerland & C. Schmit,
  Phys. Rev. E **59** (1999) R1315 — rank-one perturbations and intermediate
  statistics (#87).
- P. Sarnak, *Arithmetic quantum chaos* (Schur lectures, 1995) — the Hecke
  explanation of the arithmetic exception (predecessor headline).
- G. Sierra, arXiv:1601.01797 — the `xp`-program review already in
  [`bibliography.md`](bibliography.md)'s orbit (per-note source).
- T. Kottos & U. Smilansky, PRL **79** (1997) 4794 — quantum graphs (the
  out-of-charter calibration aside).
