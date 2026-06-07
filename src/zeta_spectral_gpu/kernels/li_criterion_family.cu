// Hand-written CUDA C kernel for the family GRH Li-coefficient sweep (issue #71).
//
// Loaded from Python via CuPy RawModule (NVRTC JIT) in li_criterion_family_gpu.py.
// One CUDA block per Dirichlet character: the family is embarrassingly parallel
// (each character is an independent, zero-free forward computation), so a whole
// family of hundreds of characters maps to a grid of hundreds of blocks. This is
// the GPU "scale" leverage for the Phase-2 Li sweep -- the fp64 *assembly*; the
// precision-delicate analytic inputs (the L-Taylor / generalized-Stieltjes data)
// are computed in mpmath on the host and passed in (see the module docstring).
//
// Per block, in fp64, for character c with normalized L-Taylor coefficients
// d_k = c_k / c_0  (k = 1..N, complex; d_0 = 1 implicit):
//   1. b_k = [u^k] log L(1+u, chi) via the log-power-series recurrence
//        b_k = d_k - (1/k) sum_{i=1}^{k-1} i * b_i * d_{k-i}        (b_0 = 0);
//   2. a_k = b_k + 2^{-k} psi^{(k-1)}(arg)/k! + (k==1 ? (log(q/pi))/2 : 0),
//      with arg = (1+parity)/2 -> the completed log Lambda(1+u, chi) coefficients
//      (the parity-shared polygamma terms `poly` and per-character log(q/pi) are
//      passed in);
//   3. lambda_n = n * sum_{j=0}^{n-1} C(n-1,j) a_{n-j}   (Bombieri-Lagarias);
//   4. reduce to min_n Re(lambda_n) and the all-Re-positive (GRH) flag.
//
// Correctness is pinned by tests/test_li_criterion_family.py, which compares this
// kernel against the pure-mpmath reference in li_criterion_family.py on small N
// (fp64 saturates the binomial-cancellation assembly past a few dozen coefficients
// -- the repo's recurring precision wall -- so the GPU path is the small-N family
// producer; deeper N stays mpmath's job).
//
// (ASCII only: CuPy re-emits this source through the Windows cp1252 codec.)

extern "C" {

__global__ void family_li(
    const double* __restrict__ d_re,    // [num_chars * N] normalized L-Taylor d_k, k=1..N
    const double* __restrict__ d_im,
    const double* __restrict__ logqpi,  // [num_chars] log(q/pi) per character
    const int* __restrict__ parity,     // [num_chars] 0 (even) / 1 (odd)
    const double* __restrict__ poly,    // [2 * N] poly[a*N + (k-1)] = psi^{(k-1)}(arg_a)/(2^k k!)
    const double* __restrict__ binom,   // [N * N] binom[(n-1)*N + j] = C(n-1, j), else 0
    const int N,
    const int num_chars,
    double* __restrict__ lam_re,        // [num_chars * N] lambda_n, n=1..N
    double* __restrict__ lam_im,
    double* __restrict__ min_re,        // [num_chars] min_n Re(lambda_n)
    int* __restrict__ all_pos)          // [num_chars] 1 if every Re(lambda_n) > 0
{
    const int c = blockIdx.x;
    if (c >= num_chars) return;

    // Shared: the completed log-Lambda coefficients a_k (k = 0..N), real and imag.
    extern __shared__ double sh[];
    double* are = sh;             // length N+1
    double* aim = sh + (N + 1);   // length N+1

    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const int base = c * N;

    // Step 1+2: sequential recurrence (single thread; parallelism is across blocks).
    if (t == 0) {
        are[0] = 0.0;
        aim[0] = 0.0;
        for (int k = 1; k <= N; ++k) {
            double accr = 0.0, acci = 0.0;
            for (int i = 1; i < k; ++i) {
                const double br = are[i], bi = aim[i];          // b_i (gamma/qpi not yet added)
                const double dr = d_re[base + (k - i - 1)];     // d_{k-i}
                const double di = d_im[base + (k - i - 1)];
                const double pr = br * dr - bi * di;            // b_i * d_{k-i}
                const double pi_ = br * di + bi * dr;
                accr += i * pr;
                acci += i * pi_;
            }
            are[k] = d_re[base + (k - 1)] - accr / k;           // b_k
            aim[k] = d_im[base + (k - 1)] - acci / k;
        }
        const int a = parity[c];
        const double lqp = logqpi[c];
        for (int k = 1; k <= N; ++k) {
            are[k] += poly[a * N + (k - 1)];                    // + gamma factor (real)
            if (k == 1) are[k] += 0.5 * lqp;                    // + (q/pi) factor (real)
        }
    }
    __syncthreads();

    // Step 3: Bombieri-Lagarias assembly, one (or more) lambda_n per thread.
    for (int n = t + 1; n <= N; n += nt) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < n; ++j) {
            const double b = binom[(n - 1) * N + j];
            sr += b * are[n - j];
            si += b * aim[n - j];
        }
        lam_re[base + (n - 1)] = n * sr;
        lam_im[base + (n - 1)] = n * si;
    }
    __syncthreads();

    // Step 4: per-character reduction (single thread; N is small).
    if (t == 0) {
        double mn = lam_re[base];
        int ap = 1;
        for (int n = 1; n <= N; ++n) {
            const double v = lam_re[base + (n - 1)];
            if (v < mn) mn = v;
            if (!(v > 0.0)) ap = 0;
        }
        min_re[c] = mn;
        all_pos[c] = ap;
    }
}

}  // extern "C"
