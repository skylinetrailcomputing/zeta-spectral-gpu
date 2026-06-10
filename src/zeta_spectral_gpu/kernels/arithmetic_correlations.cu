// Hand-written CUDA C kernel for the zero Fourier statistic (issue #84).
//
// S(u) = sum_n w_n e^{i u tau_n} over a windowed set of zero ordinates: the
// empirical side of the spectral form factor, whose prime-power peaks are the
// arithmetic content beyond GUE. Loaded via CuPy RawModule (NVRTC JIT) in
// arithmetic_correlations_gpu.py; correctness is pinned by
// tests/test_arithmetic_correlations_gpu.py against the numpy reference.
//
// One block per frequency u[k]; threads stride the zeros and the partial
// cosine/sine sums tree-reduce in shared memory. The wrapper launches with
// blockDim.x == BLOCK. (ASCII only: CuPy re-emits this source through the
// Windows cp1252 codec. Array/count params are long long: device `long` is
// 32-bit on Windows.)

#define BLOCK 256

extern "C" {

__global__ void zero_fourier(
    const double* __restrict__ tau,  // windowed zero ordinates
    const double* __restrict__ w,    // taper weights, same length
    const long long n,               // number of ordinates
    const double* __restrict__ u,    // frequency grid
    const long long m,               // number of frequencies
    double* __restrict__ out_re,     // length m
    double* __restrict__ out_im)     // length m
{
    __shared__ double sre[BLOCK];
    __shared__ double sim[BLOCK];

    const long long k = blockIdx.x;
    if (k >= m) return;
    const double uk = u[k];

    double re = 0.0, im = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        double s, c;
        sincos(uk * tau[i], &s, &c);
        re += w[i] * c;
        im += w[i] * s;
    }
    sre[threadIdx.x] = re;
    sim[threadIdx.x] = im;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sre[threadIdx.x] += sre[threadIdx.x + stride];
            sim[threadIdx.x] += sim[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_re[k] = sre[0];
        out_im[k] = sim[0];
    }
}

}  // extern "C"
