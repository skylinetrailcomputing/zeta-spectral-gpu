"""Profile the *actual* CCM closed-form pipeline phase-by-phase (issue #17).

The #17 seed doc's wall-time split ("archimedean ~80%, eigensolve ~17%") was
measured against ``connes-cvs``, which evaluates digamma at ~2.2M tanh-sinh
quadrature nodes. Our ``ccm.py`` does NOT use quadrature for the production fill
— it uses closed forms (hyp2f1 / digamma / polygamma / a geometric Lerch sum), so
the special-function count is O(N), not O(N * nodes). This script measures OUR
split before optimizing anything, per CLAUDE.md ("check the budget first").

Run: ``uv run --extra oracle python scripts/profile_ccm_phases.py``
"""

from __future__ import annotations

import time

import mpmath as mp

from zeta_spectral_gpu import ccm


def _time(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<28} {dt:8.2f} s")
    return out, dt


def profile(N: int, lam, dps: int) -> None:
    print(
        f"\n=== N={N}  lam={lam}  dps={dps}  (dim={2 * N + 1})  "
        f"mpmath backend={mp.libmp.BACKEND} ==="
    )
    with mp.workdps(dps):
        lam_m = mp.mpf(lam)
        L = 2 * mp.log(lam_m)

        A, t_asm = _time(
            "assemble_weil_matrix", lambda: ccm.assemble_weil_matrix(N, lam_m)
        )
        mode, t_eig = _time(
            "smallest_even_eigenvector", lambda: ccm.smallest_even_eigenvector(A, N)
        )
        _, t_root = _time(
            "operator_eigenvalues",
            lambda: ccm.operator_eigenvalues(mode.eigenvector, N, L, N),
        )

    total = t_asm + t_eig + t_root
    print(f"  {'-' * 38}")
    print(f"  {'TOTAL':<28} {total:8.2f} s")
    for label, dt in (
        ("assembly (special fns)", t_asm),
        ("eigensolve (inverse iter)", t_eig),
        ("root extraction", t_root),
    ):
        print(f"    {label:<26} {100 * dt / total:5.1f} %")


if __name__ == "__main__":
    # Quick representative read first, then the headline §6 cell.
    profile(N=48, lam=mp.sqrt(13), dps=120)
    profile(N=120, lam=mp.sqrt(13), dps=210)
