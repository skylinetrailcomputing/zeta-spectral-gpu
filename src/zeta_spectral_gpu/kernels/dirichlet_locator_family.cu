// Hand-written CUDA C kernels for the batched-family Dirichlet-L locator (issue #68).
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in
// dirichlet_locator_family_gpu.py. The single-character locator lives in
// kernels/dirac_mirror.cu (mobius_locator / weighted_locator): one thread per
// E-grid point, inner loop over k. This generalises it to a whole **family** of
// L-functions in one launch -- the GPU "scale" leverage the Katz-Sarnak survey
// (F1) and #51 called out: hundreds of independent, cheap fp64 scans.
//
// The forward object per member is unchanged (Sierra arXiv:1404.4252 eq. 13.6):
//
//     M'_z(E) = sum_{k<=n} chi(k) mu(k) k^{-(sigma + iE)}
//             = sum_{k<=n} chi(k) (mu(k) k^{-sigma}) exp(-i E log k),
//
// whose |M'_z| peaks at the ordinates of the zeros of L(s, chi). The character chi
// (its modulus is the only number-theoretic input) goes in and the zeros come out;
// no zero is ever consumed.
//
// Layout. Characters in a family have different periods q = |d|, so rather than
// duplicating the shared mu(k) k^{-sigma} / log k arrays per member we pass a
// **packed character table** plus per-member offsets/periods (the issue's sketch):
//
//   chi[offsets[m] + r] = chi_m(r),  r = 0..periods[m]-1   (period-q residue table)
//   mu_amp[k-1] = mu(k) k^{-sigma},  logk[k-1] = log k     (k = 1..n, shared)
//
// A thread owns one (member m, energy E[e]) pair and loops over k, reading its
// member's character by residue. The residue r = k mod q is carried incrementally
// (a compare+subtract, no per-term integer modulo). Within a warp m and k are
// uniform (only e varies), so chi/mu_amp/logk are warp-uniform, cache-friendly
// reads -- exactly the access pattern of the single-character kernels.
//
// fp64 throughout (the locator is an O(1), cancellation-free regime, so double
// precision reproduces the CPU reference to floating-point tolerance -- the house
// GPU-vs-CPU rule, pinned by tests/test_dirichlet_locator_family.py).
//
// Counts are `long long` (matches host np.int64): on Windows device `long` is
// 32-bit, which silently mis-marshals array/count params -- the riemann_siegel.cu
// lesson. Per-member offsets/periods are small and stay `int`.
//
// (ASCII only: CuPy re-emits this source through the Windows cp1252 codec, so a
// stray non-ASCII byte breaks every kernel in the file.)

extern "C" {

// Real packed characters (zeta, principal/quadratic -- the symplectic Katz-Sarnak
// family): chi is a single real table, the fast path. exp(-i E logk) =
// cos(E logk) - i sin(E logk); with x = -E logk a single sincos gives c = cos,
// s = -sin, so the term chi*mu_amp contributes (chi*mu_amp)*c to Re and
// (chi*mu_amp)*s to Im.
__global__ void family_mobius_locator(
    const double* __restrict__ chi,     // packed real char tables: chi[off[m]+r]
    const int* __restrict__ offsets,    // [num_members] start of member m's table
    const int* __restrict__ periods,    // [num_members] period q_m = |d|
    const double* __restrict__ mu_amp,  // [n] mu(k) k^{-sigma}, k=1..n (shared)
    const double* __restrict__ logk,    // [n] log k, k=1..n (shared)
    const long long n,
    const double* __restrict__ E,       // [n_e] energy grid (shared)
    const long long n_e,
    const long long num_members,
    double* __restrict__ out_re,        // [num_members * n_e] Re M'_z, row-major (m, e)
    double* __restrict__ out_im)        // [num_members * n_e] Im M'_z
{
    const long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long m = blockIdx.y;
    if (e >= n_e || m >= num_members) return;

    const double Ee = E[e];
    const int q = periods[m];
    const int off = offsets[m];
    int r = 1 % q;                       // residue of k = 1
    double sr = 0.0, si = 0.0;
    for (long long k = 0; k < n; ++k) {
        const double cw = chi[off + r] * mu_amp[k];   // chi(k+1) mu(k+1) (k+1)^{-sigma}
        double s, c;
        sincos(-Ee * logk[k], &s, &c);
        sr += cw * c;
        si += cw * s;
        if (++r == q) r = 0;
    }
    const long long idx = m * n_e + e;
    out_re[idx] = sr;
    out_im[idx] = si;
}

// Complex packed characters (genuinely complex Dirichlet chi -- higher-order /
// unitary families): chi = chi_re + i chi_im. With amp = chi * mu_amp (mu_amp
// real) and the same c = cos(E logk), s = -sin(E logk):
//   Re += chi_re*mu_amp*c - chi_im*mu_amp*s,   Im += chi_re*mu_amp*s + chi_im*mu_amp*c
// (reduces to family_mobius_locator when chi_im = 0).
__global__ void family_weighted_locator(
    const double* __restrict__ chi_re,  // packed Re char tables: chi_re[off[m]+r]
    const double* __restrict__ chi_im,  // packed Im char tables
    const int* __restrict__ offsets,    // [num_members]
    const int* __restrict__ periods,    // [num_members]
    const double* __restrict__ mu_amp,  // [n] mu(k) k^{-sigma}, k=1..n (shared)
    const double* __restrict__ logk,    // [n] log k, k=1..n (shared)
    const long long n,
    const double* __restrict__ E,       // [n_e]
    const long long n_e,
    const long long num_members,
    double* __restrict__ out_re,        // [num_members * n_e]
    double* __restrict__ out_im)
{
    const long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long m = blockIdx.y;
    if (e >= n_e || m >= num_members) return;

    const double Ee = E[e];
    const int q = periods[m];
    const int off = offsets[m];
    int r = 1 % q;
    double sr = 0.0, si = 0.0;
    for (long long k = 0; k < n; ++k) {
        const double ma = mu_amp[k];
        const double ar = chi_re[off + r] * ma;
        const double ai = chi_im[off + r] * ma;
        double s, c;
        sincos(-Ee * logk[k], &s, &c);
        sr += ar * c - ai * s;
        si += ar * s + ai * c;
        if (++r == q) r = 0;
    }
    const long long idx = m * n_e + e;
    out_re[idx] = sr;
    out_im[idx] = si;
}

}  // extern "C"
