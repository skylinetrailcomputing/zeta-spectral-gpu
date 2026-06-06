// Hand-written CUDA C kernel for Sierra's prime-driven Moebius-mirror locator.
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in dirac_mirror_gpu.py. Like
// the spacing kernels this is the deliberate "learn CUDA" surface: the eigensolves
// stay in cuSOLVER, but this embarrassingly-parallel reduction is hand-rolled.
//
// Correctness is pinned by tests/test_dirac_mirror_gpu.py, which compares the
// output against the pure-numpy reference dirac_mirror.mobius_partial_sum on a
// grid (the house GPU-vs-CPU rule).
//
// (ASCII only: CuPy re-emits this source through the Windows cp1252 codec, so a
// stray non-ASCII byte breaks every kernel in the file.)

extern "C" {

// Forward locator (Sierra arXiv:1404.4252, eq. 12.20): for each energy E[e] on the
// grid, accumulate the Moebius-Dirichlet partial sum
//
//     M'_z(n) = sum_{k<=n} amp[k] * exp(-i E logk[k]),   amp[k] = c(k) k^{-sigma},
//
// where amp[k] and logk[k] = log(k+1) are precomputed on the host. One thread per
// E-grid point loops over all k; the (n_e x n) phase matrix is never materialized,
// so the cost is O(n_e * n) FLOPs in O(n_e + n) memory -- the whole point of the
// GPU path versus the numpy outer-product reference. amp/logk are read in the same
// order by every thread (a warp-uniform, cache-friendly access pattern).
//
// exp(-i E logk) = cos(E logk) - i sin(E logk) = cos(-E logk) + i sin(-E logk),
// so with x = -E logk a single sincos gives the real (cos) and imag (sin) parts.
__global__ void mobius_locator(
    const double* __restrict__ amp,   // length n: c(k) k^{-sigma}
    const double* __restrict__ logk,  // length n: log k
    const long n,
    const double* __restrict__ E,     // length n_e: energy grid
    const long n_e,
    double* __restrict__ out_re,      // length n_e: Re M'_z
    double* __restrict__ out_im)      // length n_e: Im M'_z
{
    const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (e >= n_e) return;

    const double Ee = E[e];
    double sr = 0.0;
    double si = 0.0;
    for (long k = 0; k < n; ++k) {
        double s, c;
        sincos(-Ee * logk[k], &s, &c);   // c = cos(E logk), s = -sin(E logk)
        const double a = amp[k];
        sr += a * c;
        si += a * s;
    }
    out_re[e] = sr;
    out_im[e] = si;
}

}  // extern "C"
