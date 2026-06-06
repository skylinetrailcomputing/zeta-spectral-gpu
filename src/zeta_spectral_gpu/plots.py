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


def pair_correlation_figure(
    hist: np.ndarray,
    bin_width: float,
    *,
    n_levels: int,
    out_path: Path | str,
    title: str | None = None,
    label: str = "Riemann zeros",
) -> Path:
    """Save empirical pair correlation R2(r) vs Montgomery's GUE sine-kernel form.

    ``hist`` is the forward pair-separation histogram (CPU or GPU); ``n_levels``
    is how many unfolded levels produced it. Returns the path written.
    """
    centres, r2 = spacing.pair_correlation_density(hist, bin_width, n_levels)
    grid = np.linspace(0.0, float(centres[-1] + 0.5 * bin_width), 600)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        centres,
        r2,
        color="#5b9bd5",
        lw=1.4,
        marker="o",
        ms=3,
        label=f"{label} (N={n_levels:,})",
    )
    ax.plot(
        grid,
        spacing.montgomery_pair_correlation(grid),
        color="#c0392b",
        lw=2,
        label=r"Montgomery $1-\left(\frac{\sin \pi r}{\pi r}\right)^2$",
    )
    ax.set_xlabel("normalised separation $r$")
    ax.set_ylabel("pair correlation $R_2(r)$")
    ax.set_xlim(0.0, float(centres[-1] + 0.5 * bin_width))
    ax.set_ylim(0.0, 1.35)
    ax.set_title(title or "Zero pair correlation vs the GUE sine kernel")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
