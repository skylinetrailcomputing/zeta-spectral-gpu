# ccm-fill-precision.md — the precision anatomy of the Weil-fill, and why a double-double fill is a no (#54)

Spike #54 asked: can a **double-double** (~fp128) CUDA fill of the flagship Weil matrix
extend the usable assembly precision enough to matter — *without* taking on a full
extended-precision eigensolve? Answer: **no**, on two independent grounds. This note
records the measurement and the reasoning so the lever isn't re-litigated.

The fill in question is `ccm_gpu.assemble_weil_matrix_fp64` (the CPU reference) and its
CUDA twin `kernels/ccm_assembly.cu`: the `O(N²)` divided-difference assembly
`(B[j] − B[i]) / (π(n − m))` plus the point mass, one thread per entry. The per-mode
coefficients `B`, `a`, `diag` are computed on the host in fp64 (closed-form
Lerch / digamma / trigamma, `_per_mode_arrays_fp64`); the kernel only fills. The
multiprecision truth is `ccm.assemble_weil_matrix` (mpmath). The operator itself is
specified in [`ccm-operator.md`](ccm-operator.md).

## Where the fp64 error actually lives

Decompose the fp64 matrix error against the mpmath truth by inserting a **hybrid** — the
same fp64 coefficients, but with the divided-difference fill re-done in mpmath:

| matrix | coefficients | fill arithmetic | isolates |
|---|---|---|---|
| `A_mp` (truth) | mpmath | mpmath | — |
| `A_64` | fp64 | fp64 | total fp64 error |
| `A_hyb` (hybrid) | fp64 | mpmath | — |

Then `A_hyb − A_mp` is the error from the fp64 **coefficients** alone (fill exact), and
`A_64 − A_hyb` is the error from the fp64 **fill arithmetic** alone (same coefficients).
Relative Frobenius error, `λ = √13`, dps-100 truth:

| | N = 30 | N = 60 |
|---|---|---|
| total (fp64 vs truth) | 4.6e-15 | 8.0e-15 |
| ⤷ from fp64 coefficients | 4.6e-15 | 8.0e-15 |
| ⤷ from fp64 fill arithmetic | 5.6e-17 | 6.0e-17 |
| fill / coeff | **1.2 %** | **0.7 %** |
| near-band \|n−m\| = 1: total | 8.3e-15 | 2.6e-14 |
| near-band \|n−m\| = 1: fill | 1.1e-16 | 1.3e-16 |

**~99 % of the fp64 matrix error is the special-function coefficients, not the fill.**
The near-band rows are the tell: where the divided difference subtracts two nearly-equal
`B`'s — the catastrophic-cancellation hot spot one would expect higher precision to
rescue — the fill contributes ~1e-16 against a ~1e-14 coefficient error. That is
**Sterbenz's lemma**: subtracting two fp64 numbers within a factor of two of each other is
*exact*, so the dangerous subtraction introduces no rounding of its own. A near-band entry
is only as good as the `B`'s fed into it — a coefficient question, not a fill question.

## Verdict: negative on two independent grounds

1. **Wrong place.** A double-double fill over fp64 coefficients removes only the ~1 % fill
   slice: the matrix goes from ~15-digit to ~15-digit accuracy. ~0 digits gained. The
   precision lives in the coefficient sweep (digamma / Lerch / trigamma), which is *not*
   what "assembly fill" scopes. To gain real digits you must lift the **coefficients** to
   extended precision — a different, and much larger, build.

2. **Wrong tier, and no consumer for the extra digits.** Even a fully double-double
   assembly (coefficients included) is ~32 decimal digits — below the ~80 working digits
   needed to resolve `λ_min` through cancellation at the *smallest* cutoff `c = 13` (the
   quantity of interest is already ~1e-55 there; quad-double's ~64 digits only *barely*
   reaches it, with no headroom). And the only sink for a more-accurate matrix is an
   eigensolve: cuSOLVER `eigh` is fp64 (faithful to ~1e-13, the `conditioning_fp64`
   floor), and #54 explicitly excludes building a double-double / bignum eigensolve. The
   fp64 fill is *already more accurate (15 digits) than its consumer uses (13)*, so
   additional fill precision is unobservable downstream.

## The real lever (where a future precision push should go)

Not the fill. The precision-critical, GPU-amenable piece is the **special-function
sweep** — the per-mode `B` / `diag` coefficients — at quad-double or true fixed-precision
bignum (CGBN / campary-style limb arithmetic for the complex digamma / Lerch). And even
that only pays off **paired with an extended-precision eigensolve**, which this repo
deliberately keeps on CPU/mpmath (the `ccm.py` ↔ `ccm_gpu.py` split): the eigensolve, not
the fill, is what bounds `λ_min`. A double-double fill in isolation is Amdahl-capped from
both sides. ([`ccm-universality.md`](ccm-universality.md) is the forward science such a
path would serve; this note is only about the precision plumbing.)

## Forward?

Neutral — a tooling spike, not an experiment. It changes only *how precisely* a forward
operator is assembled, never what its spectrum is compared against. The outcome is a
documented "no": the fill is not where the fp64 wall is, so don't spend a hand-written
double-double kernel on it.

## How it was measured

The hybrid is `_per_mode_arrays_fp64(N, λ)`'s fp64 coefficients cast to `mpf`, run through
the same point-mass + divided-difference arithmetic as `assemble_weil_matrix_fp64` but
evaluated in mpmath; `A_mp` is `ccm.assemble_weil_matrix` at dps 100. Reported as per-entry
differences, relative-Frobenius over the whole matrix and over the `|n−m| = 1` band. The
measurement was a throwaway script — no production code was added, since the negative needs
none.
