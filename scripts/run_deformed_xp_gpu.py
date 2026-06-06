"""Warm-up #31: GPU dense eigensolve of Sierra's deformed-xp operator.

The GPU half of #23. Assembles the integro-differential operator (Sierra eq. 10,
arXiv:1102.5356) as a dense Hermitian matrix in a decaying scaled-Laguerre
Galerkin basis with the eq. 11 theta-extension imposed, diagonalises it with
``cupy.linalg.eigh`` (cuSOLVER), and checks the house GPU-vs-CPU rule: the low
eigenvalues reproduce the CPU secular reference ``deformed_xp.secular_spectrum``
(the real Bessel-K roots, #23) to floating-point precision, with no spurious
particle-in-a-box modes below the classical floor ``|E| >= 2h = 4*pi``.

The decaying basis is what removes the box modes a naive grid produces, and an
accurate inner Gauss-Laguerre rule for the nonlocal tail is what makes the
convergence spectral (machine-precision for the resolved low modes, then a sharp
cliff that recedes as the basis grows).

Forward, not inverse: a geometric deformation of xp — no primes, no zeros
consumed; the secular roots are only an output comparison.

    uv run python scripts/run_deformed_xp_gpu.py
    uv run python scripts/run_deformed_xp_gpu.py --n 120 --compare 16 --no-gpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import deformed_xp_gpu as gpu
from zeta_spectral_gpu import plots

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=100, help="Galerkin basis size")
    ap.add_argument("--theta", type=float, default=dxp.THETA_RIEMANN)
    ap.add_argument("--basis-scale", type=float, default=None, help="Laguerre rate c")
    ap.add_argument("--n-quad", type=int, default=None, help="outer quadrature order")
    ap.add_argument("--n-inner", type=int, default=80, help="inner (tail) quad order")
    ap.add_argument("--compare", type=int, default=12, help="levels compared to #23")
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU eigh path")
    ap.add_argument(
        "--out", type=Path, default=DATA / "deformed_xp_gpu_convergence.png"
    )
    args = ap.parse_args()

    print(
        f"deformed-xp GPU eigensolve  (N={args.n}, theta={args.theta / np.pi:.4g}*pi, "
        f"c={(args.basis_scale or 4 * gpu.L_P) / np.pi:.3g}*pi)"
    )
    t0 = time.perf_counter()
    h = gpu.galerkin_matrix(
        args.n,
        theta=args.theta,
        basis_scale=args.basis_scale,
        n_quad=args.n_quad,
        n_inner=args.n_inner,
    )
    herm = float(np.max(np.abs(h - h.conj().T)))
    print(
        f"  assembled {h.shape[0]}x{h.shape[0]} Hermitian matrix in "
        f"{time.perf_counter() - t0:.2f}s  (max|H-H^H|={herm:.1e})"
    )

    # GPU eigh (cuSOLVER), cross-checked against the CPU reference (house rule).
    if args.no_gpu:
        eig = gpu.eigenvalues(h, use_gpu=False)
        print("  eigensolve: CPU numpy.linalg.eigvalsh (forced)")
    else:
        eig = gpu.eigenvalues(h, use_gpu=True)
        eig_cpu = gpu.eigenvalues(h, use_gpu=False)
        agree = float(np.max(np.abs(np.sort(eig) - np.sort(eig_cpu))))
        print(
            f"  eigensolve: GPU cupy.linalg.eigh; GPU-vs-CPU max|dlambda|={agree:.2e}"
        )

    spec = np.sort(eig[eig > 1e-6])
    ref = dxp.secular_spectrum(args.compare, cache_dir=None)
    k = min(args.compare, spec.size)
    floor_ok = bool(np.all(spec > dxp.CLASSICAL_BOUND - 1e-6))
    print(
        f"\n  lowest eigenvalue {spec[0]:.4f} > classical floor "
        f"{dxp.CLASSICAL_BOUND:.4f}: {floor_ok}"
    )
    print("\n   k   eig_k         secular_k      |error|")
    for i in range(k):
        print(
            f"  {i + 1:2d}  {spec[i]:12.7f}  {ref[i]:12.7f}  {abs(spec[i] - ref[i]):.2e}"
        )

    # Convergence vs basis size for the figure (and as a visible demonstration).
    errors_by_n = {}
    ref5 = dxp.secular_spectrum(min(args.compare, 10), cache_dir=None)
    for nb in sorted({args.n // 2, 3 * args.n // 4, args.n}):
        sp = gpu.spectrum(nb, theta=args.theta, use_gpu=False)
        errors_by_n[nb] = np.abs(sp[: ref5.size] - ref5)
    out = plots.deformed_xp_eigh_convergence_figure(errors_by_n, out_path=args.out)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
