"""The warm-up figures render to a real PNG, headless and without a GPU."""

from __future__ import annotations

import numpy as np

from zeta_spectral_gpu import plots

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_spacing_histogram_figure_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    spacings = np.abs(rng.normal(1.0, 0.4, size=5000))  # positive, mean ~1
    out = plots.spacing_histogram_figure(
        spacings, out_path=tmp_path / "fig.png", n_bins=40
    )
    assert out.exists()
    data = out.read_bytes()
    assert data[:8] == _PNG_MAGIC
    assert len(data) > 1000  # a real figure, not an empty canvas


def test_spacing_histogram_figure_creates_parent_dirs(tmp_path):
    spacings = np.abs(np.sin(np.arange(1000))) + 0.1
    out = plots.spacing_histogram_figure(
        spacings, out_path=tmp_path / "sub" / "deep" / "f.png"
    )
    assert out.exists()


def test_spacing_ratio_figure_writes_png(tmp_path):
    from zeta_spectral_gpu import spacing

    rng = np.random.default_rng(0)
    levels = np.sort(rng.uniform(0.0, 5000.0, size=5000))
    ratios = spacing.spacing_ratios(levels)
    out = plots.spacing_ratio_figure(ratios, out_path=tmp_path / "ratio.png", n_bins=40)
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_pair_correlation_figure_writes_png(tmp_path):
    bin_width, n_levels, n_bins = 0.05, 100_000, 60
    centres = (np.arange(n_bins) + 0.5) * bin_width
    from zeta_spectral_gpu import spacing

    hist = n_levels * bin_width * spacing.montgomery_pair_correlation(centres)
    out = plots.pair_correlation_figure(
        hist, bin_width, n_levels=n_levels, out_path=tmp_path / "pc.png"
    )
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_rigidity_figure_writes_png(tmp_path):
    from zeta_spectral_gpu import spacing

    lengths = np.geomspace(1.0, 30.0, 20)
    sigma2 = spacing.gue_number_variance(lengths)
    delta3 = spacing.gue_delta3(lengths)
    out = plots.rigidity_figure(
        lengths, sigma2, delta3, n_levels=100_000, out_path=tmp_path / "rig.png"
    )
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_dirichlet_locator_figure_writes_png(tmp_path):
    grid = np.arange(2.0, 14.0, 0.05)
    true_zeros = np.array([6.020949, 10.243770, 12.988056])
    abs_m = np.zeros_like(grid)
    for z in true_zeros:  # narrow bumps standing in for located peaks
        abs_m += 2.0 * np.exp(-(((grid - z) / 0.12) ** 2))
    out = plots.dirichlet_locator_figure(
        grid,
        abs_m,
        true_zeros,
        true_zeros,
        out_path=tmp_path / "loc.png",
        n=4000,
        modulus=4,
        index=1,
        height=1.0,
    )
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC
    assert len(out.read_bytes()) > 1000


def test_dirichlet_locator_figure_handles_empty(tmp_path):
    # No peaks / no comparison zeros and no modulus label -> still a valid PNG.
    grid = np.linspace(1.0, 5.0, 50)
    out = plots.dirichlet_locator_figure(
        grid,
        np.abs(np.sin(grid)),
        np.array([]),
        np.array([]),
        out_path=tmp_path / "empty.png",
    )
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_ccm_rtilde_vs_cutoff_figure_writes_png(tmp_path):
    xs = [6, 9, 12, 14]
    rtilde_full = {6: 0.82, 9: 0.79, 12: 0.74, 14: 0.71}
    rtilde_low = {6: 0.70, 9: 0.65, 12: 0.62, 14: 0.62}
    # Without the push-N panel (single-axis branch).
    out = plots.ccm_rtilde_vs_cutoff_figure(
        xs, rtilde_full, rtilde_low, low_count=40, out_path=tmp_path / "rt.png"
    )
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC
    # With the push-N panel (two-axis branch).
    pushn = {"x": 14, "rtilde_by_N": {60: 0.62, 120: 0.69, 180: 0.72, 240: 0.75}}
    out2 = plots.ccm_rtilde_vs_cutoff_figure(
        xs,
        rtilde_full,
        rtilde_low,
        low_count=40,
        pushn=pushn,
        out_path=tmp_path / "rt_pushn.png",
    )
    assert out2.exists()
    assert out2.read_bytes()[:8] == _PNG_MAGIC
