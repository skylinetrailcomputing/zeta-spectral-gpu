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
