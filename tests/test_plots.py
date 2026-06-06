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
