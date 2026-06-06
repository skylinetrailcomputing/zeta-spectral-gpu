# Theory map: harmonic functions → conical functions → the critical line

The mathematical bridge the predecessor repo built, distilled to what is
load-bearing for the spectral approach in *this* repo. It traces why
$\operatorname{Re}(s) = \tfrac12$ keeps appearing — from the null space of the
Laplacian, through conical (Mehler–Fock) functions, to the Selberg / Riemann
spectral picture. Companion to [`computational-arc.md`](computational-arc.md),
which records what the repo actually computed along this path.

> Distilled from `wedgetrigfunctions202601`'s
> `knowledge/harmonic-functions-and-zeta.md`. That note in turn descended from
> the maintainer's longer design brief `harmonic-functions-to-zeta.md` — the
> same brief whose §10 is this repo's CUDA charter (see
> [`../project-framing.md`](../project-framing.md)).

---

## 1. Harmonic functions in 2D, and non-integer exponents

Laplace's equation $\nabla^2 u = 0$ in polar $(r, \theta)$ separates into
$r^{\alpha}\cos(\alpha\theta)$, $r^{\alpha}\sin(\alpha\theta)$, plus the $n=0$
cases $\log r$, the constant, and the (multivalued) angle $\theta$ itself. The
exponent $\alpha$ is harmonic for **any** real or complex value; integer $n$ is
forced only by demanding single-valuedness on a full disk. On a **wedge or
sector** the boundary conditions select non-integer $\alpha$ — the origin of
**corner singularities** like $r^{1/2}$. This is where the repo's original 2D
wedge scripts lived.

## 2. Up a dimension: the Euler radial equation and conical functions

In 3D spherical separation $u = R(r)\,Y_l^m$, the radial part obeys the **Euler
(equidimensional) equation** $r^2R'' + 2rR' - l(l+1)R = 0$, with clean power-law
roots $R = r^l$ or $r^{-(l+1)}$. The richness is angular, not radial.

Relax pole-regularity (cones, edges, scattering) and the separation constant
need not be $l(l+1)$ for integer $l$. Take

$$
l = -\tfrac12 + i\tau, \qquad \tau \in \mathbb{R}.
$$

The angular solutions become **conical (Mehler–Fock) Legendre functions**
$P_{-1/2+i\tau}(\cos\theta)$ with continuous index $\tau$, and the radial part
becomes

$$
r^{-1/2 + i\tau} = r^{-1/2}\, e^{i\tau \ln r}
$$

— a continuous frequency $\tau$ riding on $\log r$. The **Mehler–Fock
transform** plays the role Fourier series plays in the integer case.

### The unifying statement

$r^s$ is a harmonic radial solution when $s(s+1) = l(l+1)$, i.e. $s = l$ or
$s = -l-1$. Writing $s = -\tfrac12 + i\tau$ puts you on the **critical line**,
where the two roots are complex conjugates and $r^s$ oscillates
logarithmically. Integer $l$ keeps you *off* that line (real, well-separated
roots = pure power laws). **The continuous spectrum lives on it.**

| Regime | Degree | Radial behavior |
|---|---|---|
| Regular on the sphere | discrete integer $l$ | power laws $r^l,\ r^{-l-1}$ |
| Cones / edges / no pole-regularity | continuous $l = -\tfrac12 + i\tau$ | log-oscillation $r^{-1/2}e^{i\tau\ln r}$ |

## 3. Why the critical line, really: Mellin / scaling self-duality

The Euler equation is scale-invariant ($r \to \lambda r$), so its natural
transform is the **Mellin transform**, kernel $r^s$. The line
$\operatorname{Re}(s) = \tfrac12$ is exactly where Mellin transforms become
**unitary** (Plancherel): under the measure $dr/r$, the functions
$r^{-1/2+i\tau}$ are the unit-modulus characters of the multiplicative group
$\mathbb{R}_{>0}$.

The zeta functional equation $\zeta(s) \leftrightarrow \zeta(1-s)$ has its
symmetry axis at $\operatorname{Re}(s) = \tfrac12$ for the *same*
group-theoretic reason: it is the fixed line of $s \to 1-s$, and $\zeta$ is
built from a Mellin transform (of the theta function / the $\Gamma$-factor
completion). Both inherit the line $\tfrac12$ from self-duality of
multiplicative scaling. This part is **theorem-level**, not conjecture — and it
is the spine of the whole spectral program: the place where the analytic object
($\zeta$) and the geometric one (scaling-invariant harmonics) share an axis.

## 4. The four pillars of the spectral analogy

- **Selberg (proven).** On hyperbolic surfaces the Laplacian's eigenfunctions
  are built from exactly these conical Legendre functions. With spectral
  parameter $\lambda = s(1-s) = \tfrac14 + \tau^2$, $s = \tfrac12 + i\tau$, the
  **Selberg zeta function** is symmetric about $\operatorname{Re}(s) = \tfrac12$
  and its RH analog is **true and provable** — the hyperbolic Laplacian is
  genuinely self-adjoint with real spectrum, so every $\tau_n$ is real. This is
  the well-trodden bridge from harmonic analysis on the cone to a zeta-type
  function, and the structural model the predecessor's hyperbolic work was
  aiming at.
- **Hilbert–Pólya (conjectural).** The nontrivial zeros $\tfrac12 + i\tau$ are
  conjectured to be eigenvalues of some self-adjoint operator, forcing $\tau$
  real. No such operator is known *for $\zeta$ itself* — which is the whole
  reason this repo computes **forward**, structurally derived operators and
  *checks* their spectra against the zeros rather than fitting one (see
  [`../project-framing.md`](../project-framing.md)).
- **Montgomery–Dyson (empirical).** Spacings of $\zeta$ zeros match GUE
  eigenvalue spacings of large random Hermitian matrices — independent
  corroboration of the spectral picture, and the statistic the predecessor's
  GUE work measured directly (and this repo's warm-up phase scales).
- **Connes (the program's edge).** The adelic / noncommutative-geometry
  formulation makes scaling-invariance explicit by constructing a space whose
  "scaling Laplacian" has the zeros as its spectrum — the lineage this repo's
  flagship (Connes–Consani–Moscovici) sits in; see
  [`../ccm-operator.md`](../ccm-operator.md).

## 5. The payoff line for this repo

The harmonic-functions route arrives at a sharp, useful split:

- The critical line is **forced by self-adjointness / Mellin self-duality** —
  not by fitting. A genuine spectral operator puts its zeros on the line *for
  free*. That is the bar a forward construction must clear.
- But sitting on the line is only half the story. *Which* density and *which*
  fluctuations the spectrum carries is what separates a generic hyperbolic
  Laplacian from $\zeta$ — and that is exactly the question the
  [`computational-arc.md`](computational-arc.md) findings answer (and answer
  negatively for the simple geometries).
