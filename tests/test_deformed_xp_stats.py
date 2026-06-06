"""Warm-up #24: the universality statistics of the deformed-xp spectrum.

These wire the #23 spectrum into the existing #5/#6/#15 harness and assert the
*known* answer: the deformed-``xp`` operator reproduces the average zeros, so its
spectrum is a near-**picket fence** — ``P(s) -> delta(s-1)``, ``Sigma^2(L)``
bounded, ``Delta_3(L)`` flat — with none of the GUE fluctuations the real zeros
carry. The decisive check is *discrimination*: the same harness must report
picket-fence statistics for the xp spectrum and (GUE-like) rigidity for the real
zeros. If it called the xp spectrum GUE, the harness would be broken.

No new statistics code is exercised here — only the wiring of the deformed-xp
spectrum (``deformed_xp``) through ``spacing`` and ``plots``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import deformed_xp as dxp
from zeta_spectral_gpu import plots, spacing, zeros

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_stats_figure_writes_png(tmp_path):
    # The three-panel overlay renders to a real PNG, headless and GPU-free.
    rng = np.random.default_rng(0)
    lengths = np.geomspace(1.0, 10.0, 12)
    out = plots.deformed_xp_stats_figure(
        spacings_xp=1.0 + rng.normal(0.0, 0.02, 200),  # picket-like
        spacings_zeros=np.abs(rng.normal(1.0, 0.4, 500)),  # GUE-like
        lengths=lengths,
        sigma2_xp=0.05 * np.ones_like(lengths),
        sigma2_zeros=spacing.gue_number_variance(lengths),
        delta3_xp=0.08 * np.ones_like(lengths),
        delta3_zeros=spacing.gue_delta3(lengths),
        n_xp=200,
        n_zeros=500,
        out_path=tmp_path / "stats.png",
    )
    assert out.exists()
    data = out.read_bytes()
    assert data[:8] == _PNG_MAGIC
    assert len(data) > 1000  # a real figure, not an empty canvas


def test_deformed_xp_spectrum_is_picket_fence():
    # Run a small, genuine deformed-xp spectrum through the harness. The expected
    # signature is a picket fence, and crucially NOT GUE: spacings collapse onto 1,
    # the rigidity stays bounded/flat and far below the GUE references. (If any of
    # these read GUE-like, the wiring or the operator is wrong.)
    spec = dxp.secular_spectrum(24, cache_dir=None)
    unfolded = dxp.analytic_unfold(spec)
    s = spacing.nearest_neighbour_spacings(unfolded)

    # P(s) -> delta(s - 1): every spacing hugs 1, variance ~0 (GUE var ~0.18).
    assert s.var() < 0.01
    assert np.all(np.abs(s - 1.0) < 0.2)

    lengths = np.array([4.0, 6.0, 8.0])
    sigma2 = spacing.number_variance(unfolded, lengths, n_offsets=2000)
    delta3 = spacing.dyson_mehta_delta3(unfolded, lengths, n_offsets=2000)
    gue_sigma2 = spacing.gue_number_variance(lengths)
    gue_delta3 = spacing.gue_delta3(lengths)

    # Sigma^2 bounded and well under GUE (no logarithmic growth); Delta_3 flat.
    assert np.all(sigma2 < 0.2)
    assert np.all(sigma2 < 0.5 * gue_sigma2)
    assert np.all(delta3 < gue_delta3)
    assert delta3.max() - delta3.min() < 0.03  # flat, not rising in L


@pytest.mark.slow
def test_harness_discriminates_xp_from_real_zeros(tmp_path):
    # The full #24 acceptance check: the *same* harness, fed the deformed-xp
    # spectrum and the real Riemann zeros, separates picket fence from GUE.
    n = 100
    spec = dxp.secular_spectrum(n, cache_dir=None)
    w_xp = dxp.analytic_unfold(spec)
    tau = zeros.load_ordinates(n, source="mpmath", cache_dir=tmp_path)
    w_z = zeros.unfold(tau)

    s_xp = spacing.nearest_neighbour_spacings(w_xp)
    s_z = spacing.nearest_neighbour_spacings(w_z)
    lengths = np.array([6.0, 9.0])
    sig_xp = spacing.number_variance(w_xp, lengths, n_offsets=2000)
    sig_z = spacing.number_variance(w_z, lengths, n_offsets=2000)

    # xp is a picket fence; the zeros are GUE-like (clearly between picket and the
    # Poisson null of var(s)=1), and the harness separates them by a wide margin.
    assert s_xp.var() < 0.02
    assert 0.06 < s_z.var() < 0.30
    assert s_xp.var() < 0.2 * s_z.var()
    assert np.all(sig_xp < 0.5 * sig_z)
