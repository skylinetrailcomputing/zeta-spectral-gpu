# Computational arc: what the predecessor actually computed

The CPU build journey of `wedgetrigfunctions202601`, geometry by geometry, with
the findings that carry into this repo. Read [`theory-map.md`](theory-map.md)
first for *why* the critical line shows up; this note records *what happened*
when the repo tried to realize it on a computer, ending at the GUE result that
motivates this repo's whole framing.

> Distilled from that repo's `knowledge/` (`hyperbolic-disk-spectrum.md`,
> `hyperbolic-annulus-spectrum.md`, `modular-surface-resources.md`,
> `hejhal-algorithm.md`, `selberg-zeta.md`, `gue-spacing.md`) and `CLAUDE.md`.
> The original Python scripts (Python 3.9, `scipy`/`mpmath`) live in the
> predecessor repo; they are summarized, not copied, here — they are CPU
> sandboxes, not forward experiments in this repo's sense.

---

## The arc in one line

2D wedge trig zeros → 3D Euclidean cone matching → compact self-adjoint
hyperbolic **disk** / **annulus** → modular-surface **Maass spectrum** (LMFDB +
a from-scratch **Hejhal** solver) → spectral **Selberg zeta** → **GUE
nearest-neighbour spacing**. Every GitHub issue (#1–#8, epic #3) closed; the
research arc is complete.

## Stage by stage

### 1. 2D wedge trig (ruled out — kept only as a cautionary tale)

The original scripts found zeros of trig matching conditions for non-integer
wedge exponents $\nu$ with a contrast $k$ between sub-wedges ($k=1$ collapses to
$\sin(\nu\pi)$). A coordinate change $t = \operatorname{li}(\nu)$ made spacings
look "uniform at ~50% of the Riemann $1/\log\nu$" — but this is **tautological**:
since $\operatorname{li}'(x) = 1/\log x$, any spectrum uniform in $\nu$
automatically gets $1/\log\nu$ spacing in li-coordinates. The wedge spectrum is
essentially arithmetic in $\nu$, *not* zeta-like. **Lesson logged so it is never
re-derived: drop the li transform; reintroduce log structure only when it arises
organically.** This is a small, concrete instance of the forward/inverse trap —
reading structure into a rescaling you imposed yourself.

### 2. 3D Euclidean cone (no discrete critical-line spectrum)

A cone of half-angle $\alpha$ splitting $\mathbb{R}^3$ with material contrast
$k$, axisymmetric Laplace separated as $u = r^s f(\theta)$, gives a conical
matching condition $g(s) = P_s'(c)P_s(-c) + k\,P_s(c)P_s'(-c) = 0$ (with
$c = \cos\alpha$) — the same *shape* as the 2D wedge condition but with conical
functions. On the critical-line slice $s = -\tfrac12 + i\tau$ the condition has
**no sign changes** across every $(\alpha, k)$ tested. This is **expected, not a
bug**: an unbounded cone with a single interface is not a compact self-adjoint
problem — its Mellin spectrum is continuous, so nothing forces discrete zeros on
the line. The takeaway that organized everything after: **to pin discrete
critical-line eigenvalues you need genuinely discrete spectrum** — a finite
geometry, bound states, or a proper hyperbolic quotient.

### 3. Hyperbolic disk (first critical-line spectrum *by construction*)

Dirichlet Laplacian on a geodesic ball $B_R$, metric
$ds^2 = d\rho^2 + \sinh^2\!\rho\,d\theta^2$. The axisymmetric radial equation
reduces (under $x = \cosh\rho$) to **Legendre's equation**, and self-adjointness
forces $\lambda = \tfrac14 + \tau^2$ with the eigenvalue condition

$$
D(\tau) \equiv P_{-1/2+i\tau}(\cosh R) = 0.
$$

The spectral parameter is on the critical line **by construction** — no matching
condition, no contrived BC; self-adjointness of the hyperbolic Laplacian does
all the work. $D(\tau)$ is real for real $\tau$ (Mehler integral), so ordinary
real zero-finding suffices. Extending to non-axisymmetric modes
$P^m_{-1/2+i\tau}(\cosh R)$ and summing over $m$ cleanly demonstrated the
**two-term Weyl law** (area term + Dirichlet perimeter correction):
$N_{\mathrm{Weyl}}(\lambda) = \tfrac{\mathrm{Area}}{4\pi}\lambda -
\tfrac{L}{4\pi}\sqrt{\lambda} + O(1)$, with the boundary term taking a $-29\%$
leading-order error down to $-0.3\%$ at $\tau = 15$.

**But the density is wrong for $\zeta$.** The disk's Weyl density is polynomial
in $\lambda$ (linear in the axisymmetric/1D count, quadratic with all modes);
Riemann-zero density is $\sim \log(T/2\pi)/(2\pi)$. Sitting on the critical line
is necessary but nowhere near sufficient — the geometry has to supply the slow
logarithmic growth, which a bounded domain cannot.

### 4. Hyperbolic annulus (the contrast parameter, revived)

Two boundaries $R_1 < \rho < R_2$ drop the regularity-at-origin condition, so
the general axisymmetric solution uses **both** Legendre kinds
$a\,P_{-1/2+i\tau} + b\,Q_{-1/2+i\tau}$, and the spectrum is where a $2\times2$
boundary determinant vanishes. This is the **closest descendant of the original
wedge matching condition** — a contrast parameter $k$ in a matching condition
between two conical-Legendre expressions — but now in the genuinely compact,
self-adjoint regime that forces the critical line regardless of $k$. Dirichlet
spacing matched the 1D Weyl value $\pi/(R_2-R_1)$ to 4 parts in $10^4$; a Robin
sweep moved each $\tau_n$ smoothly and monotonically with $k$. **Still 1D Weyl,
still not zeta-like.** The annulus length acts as a 1D box; to get zeta density
you need **cusps**.

### 5. Modular surface (the destination: cusps → zeta-like density)

The non-compact, finite-volume modular surface
$\mathrm{PSL}_2(\mathbb{Z})\backslash\mathbb{H}$ has cusps, mixed spectrum
(Maass cusp forms + Eisenstein continuum), and Selberg-zeta zeros whose density
finally tracks Riemann-zero density. Two routes, both taken:

- **Ingest:** the full rigorous level-1 LMFDB `maass_rigor` dump — **2,202**
  spectral parameters $\tau_n$, $9.53 \le \tau \le 184.92$, as Arb-ball
  midpoints (~95–100 digits). Provenance: Lowry-Duda, Seymour-Howell, Child et
  al., *A database of rigorous Maass forms*
  ([arXiv:2502.01442](https://arxiv.org/abs/2502.01442), 2025), three
  independent rigorous computations cross-validating, with a rigorous
  **completeness** guarantee. LMFDB data is CC BY-SA 4.0 (cite by label, e.g.
  `1.0.1.1.1`).
- **Solve:** a pure-`mpmath` **Hejhal** algorithm from scratch (no
  pip-installable / pure-Python Maass solver exists — the field is
  Sage-or-bespoke-C). $K$-Bessel + horocycle DFT, cosine/sine split by parity,
  cost = $\operatorname{sgn}\det A(\tau)$. Validated against the first three
  LMFDB eigenvalues to ~7 digits; spurious zeros filtered by the Hecke relation
  $a_4 = a_2^2 - 1$.

Two non-obvious traps worth carrying forward: the LMFDB `symmetry` column is
**inverted** (`1` = odd / sin, `0` = even / cos — so the classical "first Maass
cusp form" $\tau_1 \approx 9.534$ is *odd*); and the level-1 dump is per-parity
**complete only up to $\tau \approx 99.58$** (the odd sector drops out above
that), so spacing analysis auto-truncates to $N = 611$.

### 6. Spectral Selberg zeta (the discrete-spectrum-only object)

From a truncated list $\tau_1,\dots,\tau_N$, the Hadamard product over the
discrete spectrum

$$
Z_{\mathrm{spec}}(s) \equiv \prod_{n=1}^{N}
  \left[\,1 + \frac{(s-\tfrac12)^2}{\tau_n^2}\,\right]
$$

vanishes exactly at $s = \tfrac12 \pm i\tau_n$, is normalized to
$Z_{\mathrm{spec}}(\tfrac12) = 1$, and is symmetric under $s \to 1-s$ for free
(it depends on $s$ only through $(s-\tfrac12)^2$). It deliberately drops the
Weierstrass convergence prefactor, the scattering matrix $\varphi(s)$, and the
identity/elliptic/parabolic trace-formula terms — so it is faithful **only on
the critical line for $|t| < \tau_N$**, which is all the spacing comparison
needs. (The infinite product genuinely diverges as $N\to\infty$ since
$\sum 1/\tau_n^2 \sim \sum 1/n$; the real $Z_\Gamma$ needs the primary factors.)

### 7. GUE spacing — the headline result

The payoff step works purely on **ordinate lists** (no Selberg-zeta values
evaluated): unfold each spectrum to mean spacing 1, then measure the
nearest-neighbour spacing distribution against the **Wigner GUE surmise**
$p_{\mathrm{GUE}}(s) = \tfrac{32}{\pi^2}s^2 e^{-4s^2/\pi}$ (level repulsion,
$p(0)=0$) and the **Poisson** baseline $e^{-s}$ (no repulsion, $p(0)=1$).
Kolmogorov–Smirnov distances, LMFDB level-1 Maass vs the first 2000 Riemann
zeros:

| spectrum | $N$ | KS vs GUE | KS vs Poisson | best fit |
|---|---:|---:|---:|---|
| **Riemann $\gamma_n$** | 1999 | **0.041** | 0.316 | **GUE** (by ~8×) |
| Maass pooled | 610 | 0.223 | **0.075** | **Poisson** |
| Maass even-only | 269 | 0.193 | **0.109** | **Poisson** |
| Maass odd-only | 340 | 0.178 | **0.141** | Poisson (tail-different) |

**Riemann zeros are textbook GUE** (Montgomery–Odlyzko). The
$\mathrm{PSL}_2(\mathbb{Z})$ Maass spectrum sits closer to **Poisson** in every
parity — including stunning near-degeneracies ($\Delta\tau \sim 10^{-3}$ at
$\tau \sim 75\text{–}100$, ~100–200× closer than GUE level repulsion allows).

## Why Maass is Poisson, and why it matters here: arithmetic chaos

The modular surface is classically chaotic, so the BGS conjecture *predicts*
GUE — yet it gives Poisson. The resolution (Bogomolny–Georgeot–Giannoni–Schmit
1992; Sarnak 1995): arithmetic surfaces carry an infinite algebra of **Hecke
operators** $T_p$ that commute with the Laplacian. Maass forms are joint
eigenfunctions of $\Delta$ and all $T_p$, so the spectrum behaves like a
**superposition of many independent sub-spectra** — which converges to Poisson.
The extra commuting symmetries supply "good quantum numbers" that wash out level
repulsion.

This is the single most important learning the predecessor hands to this repo:

- **Riemann zeros have *no* such decorating symmetries** — they sit on a single
  spectrum with nothing extra commuting — so they realize the GUE prediction.
  Extra symmetry ⇒ Poisson; bare self-adjoint spectrum ⇒ GUE. That contrast is
  why "does it carry the GUE fluctuations?" is a *discriminating* test, not a
  formality.
- It sharpens the forward/inverse rule (see [`../project-framing.md`](../project-framing.md)):
  matching the **mean density** is cheap and necessary; carrying the **GUE
  fluctuations** is the hard, informative part. The same negative recurs in this
  repo with the deformed-`xp` operator (mean yes, picket-fence fluctuations no —
  [`../deformed-xp.md`](../deformed-xp.md)).
- It tells you **where the fluctuations must come from**: not geometry alone
  (the disk/annulus give Weyl; arithmetic geometry over-symmetrizes to Poisson)
  but the **primes**, via the explicit formula — which is exactly the flagship's
  Connes–Consani–Moscovici quadratic form ([`../ccm-operator.md`](../ccm-operator.md))
  and the prime-driven Dirac-mirror track ([`../dirac-mirror.md`](../dirac-mirror.md)).

## What carries directly into this repo

- **The GUE warm-up baseline.** The predecessor's `gue_spacing.py` is the CPU
  reference that this repo's warm-up statistics (spacing #5, pair-correlation
  #6, rigidity #15, the `r̃` ratio #35) extend and scale on GPU.
- **Unfolding done right.** Analytic $N(T)$ (Riemann–von Mangoldt for the zeros;
  Weyl/Selberg for the surface) plus an empirical second-pass rescale to mean
  spacing 1 — the standard machinery this repo reuses.
- **Precision discipline.** `mpmath.zetazero` at 50 digits for the zeros; the
  Hejhal $\det A$ underflowing below `mp.dps ≈ 30`. An early, concrete taste of
  this repo's central precision constraint (fp64 is not enough for the low zeros
  — see `CLAUDE.md` and [`../ccm-operator.md`](../ccm-operator.md)).

## Key references

- Bogomolny, Georgeot, Giannoni, Schmit, *Chaotic billiards generated by
  arithmetic groups*, PRL **69** (1992) — first observation that
  $\mathrm{PSL}_2(\mathbb{Z})$ violates BGS GUE universality.
- Sarnak, *Arithmetic quantum chaos*, Schur lectures (1995) — the Hecke-algebra
  explanation.
- Montgomery, *The pair correlation of zeros of the zeta function* (1973);
  Dyson, *Statistical theory of the energy levels…* (1962); Odlyzko (1987/1992)
  — the GUE fingerprint of the Riemann zeros.
- Lowry-Duda, Seymour-Howell, Child et al.,
  [arXiv:2502.01442](https://arxiv.org/abs/2502.01442) (2025) — the rigorous
  LMFDB level-1 Maass dataset.
- Hejhal, *Eigenvalues of the Laplacian for Hecke Triangle Groups*, AMS Memoir
  469 (1992); Strömberg PhD thesis (Uppsala, 2005) — the Hejhal algorithm.
