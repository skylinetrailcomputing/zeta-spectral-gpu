// Hand-written CUDA C kernel for the CCM Weil-matrix fill (issue #9).
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in ccm_gpu.py. Like
// kernels/spacing.cu this is the deliberate "learn CUDA" surface: the heavy
// special-function coefficients (the per-mode arrays B, a, diag) are computed in
// fp64 on the host, and this kernel does the O(N^2) divided-difference assembly,
// one thread per matrix entry.
//
// The entry follows ccm.assemble_weil_matrix exactly (QW = W_{0,2} - W_R - sum_p
// W_p): a point-mass term minus, off the diagonal, the Lemma 5.1 divided
// difference (B[j]-B[i])/(pi (n-m)), and on the diagonal the precomputed
// diag[i]. Correctness is pinned by tests/test_ccm_gpu.py against the pure-numpy
// reference assemble_weil_matrix_fp64 (and through it the mpmath fill).
//
// Precision note: this is fp64. The matrix entries are O(1) and reproduce the
// mpmath fill to ~1e-12, but the minimal eigenvalue of the assembled matrix sits
// far below fp64 epsilon for all but the smallest cutoffs, so the extended-
// precision eigensolve stays on the CPU (see ccm_gpu.py and CLAUDE.md).

extern "C" {

// One thread per matrix entry (row-major, length dim*dim, dim = 2N+1). Index i
// is mode n = i - N, index j is mode m = j - N. Inputs:
//   B[dim]    : odd per-mode array J(n) + prime sin-sum
//   a[dim]    : point-mass coefficient 1/(L^2 + 16 pi^2 n^2)
//   diag[dim] : even diagonal array (archimedean diagonal + prime cos-sum)
__global__ void ccm_assemble(
    const double* __restrict__ B,
    const double* __restrict__ a,
    const double* __restrict__ diag,
    const long N,
    const double pref,        // 32 L sinh(L/4)^2
    const double L2,          // L^2
    const double fourpi2,     // 16 pi^2
    const double pi,
    double* __restrict__ A)   // length dim*dim, row-major
{
    const long dim = 2 * N + 1;
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;

    const long i = idx / dim;
    const long j = idx - i * dim;
    const long n = i - N;
    const long m = j - N;

    // Point-mass W_{0,2}(n,m) = pref (L^2 - 16 pi^2 n m) a_n a_m. n*m fits in a
    // long (|n|,|m| <= N) and converts exactly to double.
    const double pm = pref * (L2 - fourpi2 * (double)(n * m)) * a[i] * a[j];

    double val;
    if (i == j) {
        val = pm - diag[i];
    } else {
        // n - m = i - j is nonzero off the diagonal.
        val = pm - (B[j] - B[i]) / (pi * (double)(n - m));
    }
    A[idx] = val;
}

}  // extern "C"
