// Hand-written CUDA C kernel for the Riemann-Siegel main sum.
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in riemann_siegel_gpu.py. Like
// the dirac_mirror and spacing kernels this is the deliberate "learn CUDA" surface:
// the delicate parts (the theta phase and the asymptotic remainder) stay on the
// host in fp64; this embarrassingly-parallel reduction -- the O(sqrt(t)) dominant
// term -- is hand-rolled.
//
// Correctness is pinned by tests/test_riemann_siegel_gpu.py, which compares the
// output against the pure-numpy reference riemann_siegel.main_sum on a grid (the
// house GPU-vs-CPU rule).
//
// (ASCII only: CuPy re-emits this source through the Windows cp1252 codec, so a
// stray non-ASCII byte breaks every kernel in the file.)

extern "C" {

// Riemann-Siegel main sum: for each height t[e],
//
//     2 sum_{n=1}^{N(e)} cos(theta[e] - t[e] log n) / sqrt(n),
//
// with amp[k] = 1/sqrt(k+1) and logn[k] = log(k+1) precomputed on the host up to
// the grid-wide maximum N, and N(e) = floor(sqrt(t[e]/2pi)) the per-height term
// count. One thread per height loops over its own N(e) terms -- the (n_e x N)
// matrix is never materialized, so the cost is O(n_e * N) FLOPs in O(n_e + N)
// memory. amp/logn are read in the same order by every thread (a cache-friendly,
// warp-coherent pattern); N(e) varies slowly across a sorted height grid, so the
// loop-length divergence between neighbouring threads is mild.
//
// Integer types are `long long` (64-bit on every platform), NOT `long`: on Windows
// device code `long` is 32-bit, so a `long*` reading an int64 array mis-indexes
// every element past the first. (`long` scalars work because their low 32 bits are
// correct on little-endian -- which is why the dirac_mirror kernel, whose only
// integers are scalars, never tripped this.)
__global__ void rs_main_sum(
    const double* __restrict__ amp,    // length n_max: 1/sqrt(n), n = 1..n_max
    const double* __restrict__ logn,   // length n_max: log n
    const double* __restrict__ t,      // length n_e: heights
    const double* __restrict__ theta,  // length n_e: Riemann-Siegel theta(t)
    const long long* __restrict__ big_n,  // length n_e: N = floor(sqrt(t/2pi))
    const long long n_e,
    double* __restrict__ out)          // length n_e: the main sum
{
    const long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= n_e) return;

    const double te = t[e];
    const double th = theta[e];
    const long long n = big_n[e];
    double s = 0.0;
    for (long long k = 0; k < n; ++k) {
        s += amp[k] * cos(th - te * logn[k]);
    }
    out[e] = 2.0 * s;
}

}  // extern "C"
