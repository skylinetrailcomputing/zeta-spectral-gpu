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


def rigidity_figure(
    lengths: np.ndarray,
    sigma2: np.ndarray,
    delta3: np.ndarray,
    *,
    n_levels: int,
    out_path: Path | str,
    saturation_l: float | None = None,
    title: str | None = None,
    label: str = "Riemann zeros",
) -> Path:
    """Save the two rigidity statistics vs their GUE and Poisson references.

    Left panel: number variance ``Sigma^2(L)``; right panel: Dyson-Mehta
    ``Delta_3(L)``. Both on a log-``L`` axis, with the exact GUE curves (sine
    kernel + its Mehta transform) and the Poisson nulls (``L`` and ``L/15``).

    The zeros track GUE only up to the Berry saturation scale
    ``L* ~ ln(T/2pi) / pi`` (drawn if ``saturation_l`` is given); beyond it the
    arithmetic (prime) contributions make them saturate *below* the GUE
    logarithm — more rigid than GUE at long range. The y-axes are scaled to the
    GUE/empirical detail, so the linear Poisson nulls deliberately leave the
    frame. Returns the path written.
    """
    L = np.asarray(lengths, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    delta3 = np.asarray(delta3, dtype=np.float64)
    grid = np.geomspace(L.min(), L.max(), 400)

    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    panels = (
        (ax_v, sigma2, spacing.gue_number_variance(grid), grid, r"\Sigma^2(L)"),
        (ax_d, delta3, spacing.gue_delta3(grid), grid / 15.0, r"\Delta_3(L)"),
    )
    for ax, empirical, gue_curve, poisson_curve, sym in panels:
        ax.plot(
            L,
            empirical,
            color="#5b9bd5",
            lw=1.4,
            marker="o",
            ms=3,
            label=f"{label} (N={n_levels:,})",
        )
        ax.plot(grid, gue_curve, color="#c0392b", lw=2, label="GUE")
        ax.plot(grid, poisson_curve, color="#27ae60", lw=2, ls="--", label="Poisson")
        if saturation_l is not None and L.min() <= saturation_l <= L.max():
            ax.axvline(
                saturation_l,
                color="#7f7f7f",
                lw=1.2,
                ls=":",
                label=r"Berry $L_*\approx\ln(T/2\pi)/\pi$",
            )
        ax.set_xscale("log")
        ax.set_xlabel("window length $L$ (mean spacings)")
        # Scale to the GUE/empirical detail; the linear Poisson null runs off-frame.
        top = 1.35 * max(float(gue_curve[-1]), float(np.nanmax(empirical)))
        ax.set_ylim(0.0, top)
        ax.legend(loc="upper right", fontsize=8)

    ax_v.set_ylabel(r"number variance $\Sigma^2(L)$")
    ax_v.set_title(r"Number variance $\Sigma^2(L)$")
    ax_d.set_ylabel(r"spectral rigidity $\Delta_3(L)$")
    ax_d.set_title(r"Dyson--Mehta $\Delta_3(L)$")

    fig.suptitle(title or "Spectral rigidity of the Riemann zeros vs GUE")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def dbn_rigidity_figure(
    lengths: np.ndarray,
    sigma2_by_t: dict[float, np.ndarray],
    delta3_by_t: dict[float, np.ndarray],
    *,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save rigidity vs window length for a family of De Bruijn–Newman times ``t``.

    One curve per ``t`` in each panel (number variance ``Sigma^2(L)`` left,
    Dyson-Mehta ``Delta_3(L)`` right), shaded light (small ``t``) to dark (large
    ``t``), against the exact GUE references. The signature of the spike: as ``t``
    grows the curves fall *below* GUE toward the picket-fence floor — the zeros
    get more rigid under the heat flow. Returns the path written.
    """
    L = np.asarray(lengths, dtype=np.float64)
    grid = np.geomspace(L.min(), L.max(), 400)
    t_values = sorted(sigma2_by_t)
    cmap = plt.get_cmap("viridis")
    # Map t -> colour; guard against a single-t call (denominator 0).
    t_span = (t_values[-1] - t_values[0]) or 1.0

    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, by_t, gue_curve, sym in (
        (ax_v, sigma2_by_t, spacing.gue_number_variance(grid), r"\Sigma^2(L)"),
        (ax_d, delta3_by_t, spacing.gue_delta3(grid), r"\Delta_3(L)"),
    ):
        for t in t_values:
            ax.plot(
                L,
                np.asarray(by_t[t], dtype=np.float64),
                color=cmap(0.12 + 0.78 * (t - t_values[0]) / t_span),
                lw=1.4,
                marker="o",
                ms=3,
                label=f"$t={t:g}$",
            )
        ax.plot(grid, gue_curve, color="#c0392b", lw=2, ls="--", label="GUE")
        ax.set_xscale("log")
        ax.set_xlabel("window length $L$ (mean spacings)")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax_v.set_ylabel(r"number variance $\Sigma^2(L)$")
    ax_v.set_title(r"Number variance $\Sigma^2(L)$")
    ax_d.set_ylabel(r"spectral rigidity $\Delta_3(L)$")
    ax_d.set_title(r"Dyson--Mehta $\Delta_3(L)$")

    fig.suptitle(
        title or r"De Bruijn–Newman flow: rigidity increases with $t$ (more rigid)"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_convergence_figure(
    errors_by_lambda: dict[str, np.ndarray],
    *,
    N: int,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save the flagship convergence figure: ``|eig_k - t_k|`` vs zero index ``k``.

    One curve per cutoff ``lambda`` (labelled by ``x = lambda^2``), ``y`` on a log
    scale because the errors span tens of orders of magnitude (``~1e-55`` at
    ``k = 1`` up to ``~1e-3`` at ``k = 50`` for the headline case). The forward
    content: a *structurally derived* operator's spectrum reproducing the first
    few-dozen zeta ordinates to extreme accuracy, with the error growing with the
    index and shrinking fast as the prime cutoff ``x`` grows. Returns the path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.get_cmap("viridis")
    labels = list(errors_by_lambda)
    span = (len(labels) - 1) or 1
    for i, label in enumerate(labels):
        err = np.asarray(errors_by_lambda[label], dtype=np.float64)
        k = np.arange(1, err.size + 1)
        ax.semilogy(
            k,
            err,
            color=cmap(0.12 + 0.78 * i / span),
            lw=1.4,
            marker="o",
            ms=3,
            label=label,
        )
    ax.set_xlabel("zero index $k$")
    ax.set_ylabel(r"$|\,\mathrm{eig}_k - t_k\,|$")
    ax.set_title(title or f"CCM finite-cutoff convergence to the zeros ($N={N}$)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title="cutoff", loc="upper left")
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
