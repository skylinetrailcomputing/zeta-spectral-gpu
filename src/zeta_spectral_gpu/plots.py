"""Matplotlib figures for the warm-up statistics.

Headless (Agg backend) so figures render in CI and on a display-less server. The
figures only ever *characterise* the zeros against the random-matrix surmises —
forward, not inverse.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from . import spacing

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def spacing_histogram_figure(
    spacings: np.ndarray,
    *,
    out_path: Path | str,
    n_bins: int = 60,
    s_max: float = 4.0,
    title: str | None = None,
    label: str = "Riemann zeros",
) -> Path:
    """Save a nearest-neighbour spacing histogram vs the GUE/Poisson surmises.

    ``spacings`` are unfolded (unit-mean) nearest-neighbour spacings. Returns the
    path written. The GUE Wigner surmise is the Montgomery-Odlyzko target; Poisson
    is the uncorrelated null for contrast.
    """
    s = np.asarray(spacings, dtype=np.float64)
    centres, density = spacing.spacing_density(s, n_bins=n_bins, s_max=s_max)
    grid = np.linspace(1e-3, s_max, 400)
    width = float(centres[1] - centres[0]) if centres.size > 1 else s_max / n_bins

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(
        centres,
        density,
        width=width,
        align="center",
        color="#cfe3f7",
        edgecolor="#5b9bd5",
        label=f"{label} (N={s.size:,})",
    )
    ax.plot(grid, spacing.gue_wigner_surmise(grid), color="#c0392b", lw=2, label="GUE")
    ax.plot(
        grid,
        spacing.poisson_surmise(grid),
        color="#27ae60",
        lw=2,
        ls="--",
        label="Poisson",
    )
    ax.set_xlabel("normalised spacing $s$")
    ax.set_ylabel("probability density $p(s)$")
    ax.set_xlim(0.0, s_max)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title or "Nearest-neighbour spacing vs GUE")
    ax.legend()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
