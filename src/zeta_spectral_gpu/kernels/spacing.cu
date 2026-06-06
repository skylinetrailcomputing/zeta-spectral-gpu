// Hand-written CUDA C kernels for spacing / pair-correlation statistics.
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in spacing_gpu.py. This is
// the deliberate "learn CUDA" surface of the repo: the eigensolves stay in
// cuSOLVER, but the embarrassingly-parallel reductions are hand-rolled here.
//
// Correctness is pinned by tests/test_spacing.py, which compares the output of
// these kernels against the pure-numpy reference in spacing.py on small inputs.

extern "C" {

// Pair-correlation histogram: for each level i, scan forward over j > i and bin
// the separation (x[j] - x[i]) while it stays below max_sep. The input MUST be
// sorted ascending, which lets each thread break as soon as it leaves range —
// turning the naive O(N^2) into roughly O(N * k) for k neighbours in range.
//
// One thread per i. Histogram is accumulated with atomicAdd; callers pass a
// zero-initialised `hist` of length n_bins.
__global__ void pair_correlation_hist(
    const double* __restrict__ x,   // ascending unfolded levels
    const long n,                   // number of levels
    const double bin_width,
    const long n_bins,
    const double max_sep,
    unsigned long long* hist)       // length n_bins, zero-initialised
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    for (long j = i + 1; j < n; ++j) {
        const double d = x[j] - xi;
        if (d >= max_sep) break;            // sorted => no further hits
        const long b = (long)(d / bin_width);
        if (b >= 0 && b < n_bins) {
            atomicAdd(&hist[b], 1ULL);
        }
    }
}

// Consecutive nearest-neighbour spacings: s[i] = x[i+1] - x[i]. Trivial, but
// kept as a kernel so the GPU path has no host round-trip between steps.
__global__ void consecutive_diff(
    const double* __restrict__ x,
    const long n,                   // length of x; writes n-1 spacings
    double* __restrict__ s)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    s[i] = x[i + 1] - x[i];
}

// Folded consecutive spacing ratios r[i] = min(s_i, s_{i+1}) / max(s_i, s_{i+1}),
// with s_i = x[i+1] - x[i] (Atas 2013). Fuses the two gap differences and the
// min/max into one pass so there is no host round-trip and the raw gaps never
// leave the device. Input ascending; writes n-2 ratios. One thread per ratio.
// A zero max gap yields the same inf/nan the numpy reference produces (surfaced).
// (ASCII only: CuPy re-emits this source through the Windows cp1252 codec.)
__global__ void spacing_ratios(
    const double* __restrict__ x,
    const long n,                   // length of x; writes n-2 ratios
    double* __restrict__ r)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n - 2) return;
    const double s0 = x[i + 1] - x[i];
    const double s1 = x[i + 2] - x[i + 1];
    r[i] = fmin(s0, s1) / fmax(s0, s1);
}

}  // extern "C"
