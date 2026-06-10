"""GPU census path vs the CPU reference (issue #86) -- the house agreement rule.

The census machinery is evaluator-injectable; the GPU path is just the #55
kernel evaluator plugged into the same scan. Skips cleanly without cupy.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import lehmer_census as lc


def test_gpu_scan_reproduces_cpu_census():
    pytest.importorskip("cupy")
    from zeta_spectral_gpu import riemann_siegel_gpu as gpu

    w_cpu = lc.scan_zeros(6900.0, 7110.0)
    w_gpu = lc.scan_zeros(6900.0, 7110.0, evaluator=gpu.hardy_z_gpu)
    assert w_cpu.zeros.size == w_gpu.zeros.size
    np.testing.assert_allclose(w_gpu.zeros, w_cpu.zeros, rtol=0, atol=1e-8)

    rows_cpu = lc.lehmer_census(w_cpu)
    rows_gpu = lc.lehmer_census(w_gpu)
    assert len(rows_cpu) == len(rows_gpu)
    best_c, best_g = rows_cpu[0], rows_gpu[0]
    assert abs(best_c.delta2g - best_g.delta2g) < 1e-6
    assert abs(best_c.lam - best_g.lam) < 1e-12
