"""Matplotlib figures for the warm-up statistics.

Headless (Agg backend) so figures render in CI and on a display-less server. The
figures only ever *characterise* the zeros against the random-matrix surmises —
forward, not inverse.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from . import katz_sarnak, spacing, zeros

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


def spacing_ratio_figure(
    ratios: np.ndarray,
    *,
    out_path: Path | str,
    n_bins: int = 50,
    title: str | None = None,
    label: str = "Riemann zeros",
) -> Path:
    """Save a folded spacing-ratio histogram ``P(r̃)`` vs the GUE/GOE/Poisson surmises.

    ``ratios`` are folded consecutive ratios ``r̃ in [0, 1]`` from
    :func:`spacing.spacing_ratios` — no unfolding. The Atas surmises (GUE ``β=2``,
    GOE ``β=1``) and the Poisson null are overlaid, with the empirical ``⟨r̃⟩``
    annotated against the reference means (dotted guides). Returns the path written.
    """
    r = np.asarray(ratios, dtype=np.float64)
    r = r[np.isfinite(r)]
    centres, density = spacing.ratio_density(r, n_bins=n_bins)
    width = float(centres[1] - centres[0]) if centres.size > 1 else 1.0 / n_bins
    grid = np.linspace(1e-3, 1.0, 400)
    mean_emp = float(np.mean(r)) if r.size else float("nan")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(
        centres,
        density,
        width=width,
        align="center",
        color="#cfe3f7",
        edgecolor="#5b9bd5",
        label=rf"{label} (N={r.size:,}, $\langle\tilde r\rangle$={mean_emp:.4f})",
    )
    ax.plot(
        grid,
        spacing.folded_ratio_surmise(grid, 2),
        color="#c0392b",
        lw=2,
        label=r"GUE ($\beta=2$)",
    )
    ax.plot(
        grid,
        spacing.folded_ratio_surmise(grid, 1),
        color="#e67e22",
        lw=2,
        ls="-.",
        label=r"GOE ($\beta=1$)",
    )
    ax.plot(
        grid,
        spacing.folded_poisson_ratio_surmise(grid),
        color="#27ae60",
        lw=2,
        ls="--",
        label="Poisson",
    )
    # Reference means (dotted) and the empirical mean for at-a-glance comparison.
    for mval, mcol in (
        (spacing.MEAN_RATIO_GUE, "#c0392b"),
        (spacing.MEAN_RATIO_GOE, "#e67e22"),
        (spacing.MEAN_RATIO_POISSON, "#27ae60"),
    ):
        ax.axvline(mval, color=mcol, lw=1.0, ls=":", alpha=0.6)
    ax.axvline(
        mean_emp,
        color="#5b9bd5",
        lw=1.4,
        ls=":",
        label=r"empirical $\langle\tilde r\rangle$",
    )
    ax.set_xlabel(r"folded spacing ratio $\tilde r$")
    ax.set_ylabel(r"probability density $P(\tilde r)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title or r"Spacing ratio $\tilde r$ vs GUE (unfolding-free)")
    ax.legend(fontsize=8)
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


def ccm_precision_wall_figure(
    x_values: np.ndarray,
    eps_mpmath: np.ndarray,
    eps_fp64: np.ndarray,
    *,
    N: int,
    out_path: Path | str,
    cond_mpmath: np.ndarray | None = None,
    cond_fp64: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Save the precision-wall figure: where fp64/GPU falls off the eigensolve.

    Left panel: the minimal eigenvalue ``epsilon_N`` versus the prime cutoff ``x``
    — the extended-precision (mpmath) value plummets geometrically while the fp64
    ``eigh`` (cuSOLVER) value flattens out at the ``~1e-13`` roundoff floor. Right
    panel (optional): the condition number ``sigma_max / sigma_min``, which the
    true ``~1/epsilon_N`` sends to ``1e60+`` while fp64 saturates near ``1e16``.

    This is the empirical shadow of the limit-control problem (issue #9): the
    convergence and conditioning of ``QW_lambda^N`` are intrinsically a
    multiprecision computation, and the figure shows the exact cutoff past which
    a double-precision GPU eigensolve can no longer see the answer. Returns the
    path written.
    """
    x = np.asarray(x_values, dtype=np.float64)
    two_panel = cond_mpmath is not None and cond_fp64 is not None
    if two_panel:
        fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    else:
        fig, ax_e = plt.subplots(figsize=(7.0, 4.5))

    ax_e.axhline(
        2.2e-16,
        color="#7f7f7f",
        lw=1.0,
        ls=":",
        label=r"fp64 $\epsilon\approx2.2\!\times\!10^{-16}$",
    )
    ax_e.semilogy(
        x, eps_mpmath, color="#c0392b", lw=1.6, marker="o", ms=4, label="mpmath (exact)"
    )
    ax_e.semilogy(
        x, eps_fp64, color="#5b9bd5", lw=1.6, marker="s", ms=4, label="fp64 eigh (GPU)"
    )
    ax_e.set_xlabel("prime cutoff $x = \\lambda^2$")
    ax_e.set_ylabel(r"minimal eigenvalue $\epsilon_N$")
    ax_e.set_title(r"Convergence: $\epsilon_N \to 0$")
    ax_e.grid(True, which="both", ls=":", alpha=0.4)
    ax_e.legend(loc="upper right", fontsize=8)

    if two_panel:
        ax_c.semilogy(
            x,
            np.asarray(cond_mpmath, dtype=np.float64),
            color="#c0392b",
            lw=1.6,
            marker="o",
            ms=4,
            label="mpmath (exact)",
        )
        ax_c.semilogy(
            x,
            np.asarray(cond_fp64, dtype=np.float64),
            color="#5b9bd5",
            lw=1.6,
            marker="s",
            ms=4,
            label="fp64 eigh (GPU)",
        )
        ax_c.set_xlabel("prime cutoff $x = \\lambda^2$")
        ax_c.set_ylabel(r"condition number $\sigma_{\max}/\sigma_{\min}$")
        ax_c.set_title("Conditioning: the limit-control shadow")
        ax_c.grid(True, which="both", ls=":", alpha=0.4)
        ax_c.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        title or f"CCM precision wall: fp64 cannot reach the spectrum ($N={N}$)"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_rigidity_vs_lambda_figure(
    lengths: np.ndarray,
    sigma2_by_x: dict[float, np.ndarray],
    delta3_by_x: dict[float, np.ndarray],
    *,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save rigidity vs window length for a family of prime cutoffs ``x``.

    One curve per cutoff ``x = lambda^2`` in each panel (number variance left,
    Dyson-Mehta ``Delta_3`` right), shaded light (small ``x``) to dark (large
    ``x``), against the exact GUE references. This is the #9 forward prediction:
    the operator is built from the von Mangoldt sum over ``p <= x``, so as ``x``
    grows (more primes) the computed spectrum should *track GUE out to larger
    ``L`` before saturating* — the finite-cutoff analogue of how a higher height
    ``T`` pushes the Berry scale ``L* ~ ln(T/2pi)/pi`` outward for the real zeros.

    Finite-dimension caveat (per the #15/#20 cross-links): at ``N ~ 120`` only a
    few-dozen levels exist and the empirical unfolding suppresses the *absolute*
    rigidity, so the readout is the **cross-``x`` trend**, not the level-vs-GUE
    offset. Returns the path written.
    """
    L = np.asarray(lengths, dtype=np.float64)
    grid = np.geomspace(L.min(), L.max(), 400)
    x_values = sorted(sigma2_by_x)
    cmap = plt.get_cmap("viridis")
    x_span = (x_values[-1] - x_values[0]) or 1.0

    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, by_x, gue_curve in (
        (ax_v, sigma2_by_x, spacing.gue_number_variance(grid)),
        (ax_d, delta3_by_x, spacing.gue_delta3(grid)),
    ):
        for xv in x_values:
            ax.plot(
                L,
                np.asarray(by_x[xv], dtype=np.float64),
                color=cmap(0.12 + 0.78 * (xv - x_values[0]) / x_span),
                lw=1.4,
                marker="o",
                ms=3,
                label=f"$x={xv:g}$",
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
        title or r"CCM operator: does GUE-tracking extend with the prime cutoff $x$?"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_rtilde_vs_cutoff_figure(
    x_values,
    rtilde_full: dict,
    rtilde_low: dict,
    *,
    low_count: int,
    pushn: dict | None = None,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save the spacing-ratio rigidity readout of the CCM spectrum (#18).

    Left panel: the mean folded spacing ratio ``<r~>`` (Atas 2013) vs the prime
    cutoff ``x = lambda^2``, for the full computed spectrum and for the low
    (zero-tracking) window, against the GUE / Poisson / picket-fence references.
    ``<r~>`` is **unfolding-free**, so unlike ``Sigma^2``/``Delta_3`` it is immune to
    the finite-``N`` unfolding suppression that washed out the #9 read. The forward
    signal is the monotone descent: as ``x`` grows (more primes) the operator's local
    rigidity relaxes toward the zeros' GUE value (``0.6027``).

    Right panel (when ``pushn`` is given): the same ``<r~>`` vs the truncation ``N``
    at fixed ``x``. It drifts *away* from GUE, toward the picket-fence ``1.0``, as
    ``N`` grows — the added high-energy levels are pole-locked (the zero density
    outruns the pole spacing ``2 pi / L``), so enlarging ``N`` extends the non-GUE
    tail rather than the GUE-tracking range. The lever is the prime cutoff, not the
    matrix size (#18). Returns the path written.
    """
    xs = sorted(x_values)
    full = [rtilde_full[x] for x in xs]
    low = [rtilde_low[x] for x in xs]

    has_push = pushn is not None
    fig, axes = plt.subplots(
        1,
        2 if has_push else 1,
        figsize=(11.0 if has_push else 6.0, 4.5),
        squeeze=False,
    )
    ax = axes[0, 0]
    ax.axhline(
        spacing.MEAN_RATIO_GUE,
        color="#c0392b",
        ls="--",
        lw=2,
        label=r"GUE ($\langle\tilde r\rangle=0.6027$)",
    )
    ax.axhline(
        spacing.MEAN_RATIO_POISSON,
        color="#7f7f7f",
        ls=":",
        lw=1.5,
        label="Poisson (0.386)",
    )
    ax.axhline(1.0, color="#2c3e50", ls="-.", lw=1.0, label="picket fence (1.0)")
    ax.plot(xs, full, color="#5b9bd5", marker="o", lw=1.6, label="full spectrum")
    ax.plot(
        xs,
        low,
        color="#27ae60",
        marker="s",
        lw=1.6,
        label=f"low {low_count} (zero-tracking)",
    )
    ax.set_xlabel(r"prime cutoff $x = \lambda^2$")
    ax.set_ylabel(r"mean folded spacing ratio $\langle\tilde r\rangle$")
    ax.set_title("Local rigidity relaxes toward GUE as primes grow")
    ax.legend(fontsize=8, loc="upper right")

    if has_push:
        ax2 = axes[0, 1]
        ns = sorted(pushn["rtilde_by_N"])
        vals = [pushn["rtilde_by_N"][n] for n in ns]
        ax2.axhline(spacing.MEAN_RATIO_GUE, color="#c0392b", ls="--", lw=2, label="GUE")
        ax2.axhline(1.0, color="#2c3e50", ls="-.", lw=1.0, label="picket fence")
        ax2.plot(
            ns,
            vals,
            color="#8e44ad",
            marker="o",
            lw=1.6,
            label=rf"$x={pushn['x']:g}$, full spectrum",
        )
        ax2.set_xlabel(r"truncation $N$ (dim $2N+1$)")
        ax2.set_ylabel(r"$\langle\tilde r\rangle$")
        ax2.set_title(r"Pushing $N$ drifts away from GUE (pole-locked tail)")
        ax2.legend(fontsize=8, loc="center right")

    fig.suptitle(title or r"CCM operator: spacing-ratio rigidity vs prime cutoff")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def deformed_xp_staircase_figure(
    spectrum: np.ndarray,
    *,
    out_path: Path | str,
    average_zeros: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Save the deformed-``xp`` staircase ``N(E)`` vs the smooth zero count.

    Left: the computed counting function ``N(E) = #{E_k <= E}`` (step) against the
    analytic smooth term ``N_bar(E)`` (``zeros.smooth_count``) — the forward sanity
    check that the spectrum reproduces the *average* zero density (matching slope).
    Optional ``average_zeros`` markers show the smooth heights the levels track.
    Right: the analytic-unfolded nearest-neighbour spacings hugging 1 — the mean
    density is reproduced; whether the *fluctuations* match GUE is the separate
    question the companion statistics (#24) answer. Returns the path written.
    """
    E = np.sort(np.asarray(spectrum, dtype=np.float64))
    n = E.size
    counts = np.arange(1, n + 1)
    grid = np.linspace(float(E[0]) * 0.6, float(E[-1]) * 1.02, 400)

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax_s.step(
        E,
        counts,
        where="post",
        color="#5b9bd5",
        lw=1.6,
        label=f"deformed-$xp$ $N(E)$ (N={n})",
    )
    ax_s.plot(
        grid,
        zeros.smooth_count(grid),
        color="#c0392b",
        lw=2,
        label=r"smooth $\bar N(E)$",
    )
    if average_zeros is not None:
        az = np.sort(np.asarray(average_zeros, dtype=np.float64))
        ax_s.plot(
            az,
            np.arange(1, az.size + 1) - 0.5,
            ls="none",
            marker="v",
            ms=5,
            color="#7f7f7f",
            label="average zeros",
        )
    ax_s.set_xlabel("$E$")
    ax_s.set_ylabel("counting function $N(E)$")
    ax_s.set_title("Staircase vs smooth term")
    ax_s.legend(loc="upper left", fontsize=8)

    s = np.diff(zeros.smooth_count(E))
    ax_d.plot(
        counts[1:],
        s,
        color="#5b9bd5",
        lw=1.2,
        marker="o",
        ms=4,
        label="unfolded spacings $s_n$",
    )
    ax_d.axhline(1.0, color="#c0392b", lw=2, ls="--", label="mean $=1$")
    ax_d.set_xlabel("level index $n$")
    ax_d.set_ylabel("unfolded spacing $s_n$")
    ax_d.set_ylim(bottom=0.0)
    ax_d.set_title("Mean density reproduced (fluctuations: #24)")
    ax_d.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        title or r"Sierra deformed-$xp$: spectrum reproduces the average zeros"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def deformed_xp_stats_figure(
    *,
    spacings_xp: np.ndarray,
    spacings_zeros: np.ndarray,
    lengths: np.ndarray,
    sigma2_xp: np.ndarray,
    sigma2_zeros: np.ndarray,
    delta3_xp: np.ndarray,
    delta3_zeros: np.ndarray,
    n_xp: int,
    n_zeros: int,
    out_path: Path | str,
    n_bins: int = 40,
    s_max: float = 4.0,
    title: str | None = None,
) -> Path:
    """Overlay the deformed-``xp`` spectrum and the real zeros across all three
    warm-up statistics — the #24 payoff figure.

    Three panels share the layout "two empirical series vs the GUE and Poisson
    references": nearest-neighbour spacing ``P(s)`` (left), number variance
    ``Sigma^2(L)`` (centre) and Dyson--Mehta ``Delta_3(L)`` (right). The
    deformed-``xp`` spectrum (#23) reproduces the *average* zeros, so its statistics
    collapse to a near-**picket fence** — ``P(s)`` spikes at ``s=1`` and the rigidity
    curves stay bounded/flat — while the real zeros follow GUE. The visible gap
    between the two empirical series *is* the result: matching the mean density
    (which both do) is necessary but nowhere near sufficient for the GUE
    fluctuations. Returns the path written.
    """
    xp_colour, zero_colour = "#8e44ad", "#5b9bd5"
    L = np.asarray(lengths, dtype=np.float64)
    grid = np.geomspace(L.min(), L.max(), 400)

    fig, (ax_p, ax_v, ax_d) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # P(s): the deformed-xp spacings spike at 1 (picket fence); the zeros follow GUE.
    centres, dens_xp = spacing.spacing_density(spacings_xp, n_bins=n_bins, s_max=s_max)
    _, dens_z = spacing.spacing_density(spacings_zeros, n_bins=n_bins, s_max=s_max)
    width = float(centres[1] - centres[0]) if centres.size > 1 else s_max / n_bins
    sgrid = np.linspace(1e-3, s_max, 400)
    ax_p.bar(
        centres,
        dens_xp,
        width=width,
        color="#e7d6f0",
        edgecolor=xp_colour,
        label=f"deformed-$xp$ (N={n_xp})",
    )
    ax_p.step(
        centres,
        dens_z,
        where="mid",
        color=zero_colour,
        lw=1.7,
        label=f"real zeros (N={n_zeros:,})",
    )
    ax_p.plot(
        sgrid, spacing.gue_wigner_surmise(sgrid), color="#c0392b", lw=2, label="GUE"
    )
    ax_p.plot(
        sgrid,
        spacing.poisson_surmise(sgrid),
        color="#27ae60",
        lw=2,
        ls="--",
        label="Poisson",
    )
    ax_p.set_xlim(0.0, s_max)
    ax_p.set_ylim(bottom=0.0)
    ax_p.set_xlabel("normalised spacing $s$")
    ax_p.set_ylabel("probability density $p(s)$")
    ax_p.set_title("Nearest-neighbour spacing $P(s)$")
    ax_p.legend(fontsize=8)

    # Sigma^2(L) and Delta_3(L): both empirical series vs GUE (rigid) and Poisson.
    panels = (
        (
            ax_v,
            sigma2_xp,
            sigma2_zeros,
            spacing.gue_number_variance(grid),
            grid,
            r"number variance $\Sigma^2(L)$",
        ),
        (
            ax_d,
            delta3_xp,
            delta3_zeros,
            spacing.gue_delta3(grid),
            grid / 15.0,
            r"spectral rigidity $\Delta_3(L)$",
        ),
    )
    for ax, emp_xp, emp_z, gue_curve, poisson_curve, ylabel in panels:
        ax.plot(
            L,
            emp_xp,
            color=xp_colour,
            lw=1.4,
            marker="s",
            ms=3,
            label=f"deformed-$xp$ (N={n_xp})",
        )
        ax.plot(
            L,
            emp_z,
            color=zero_colour,
            lw=1.4,
            marker="o",
            ms=3,
            label=f"real zeros (N={n_zeros:,})",
        )
        ax.plot(grid, gue_curve, color="#c0392b", lw=2, label="GUE")
        ax.plot(grid, poisson_curve, color="#27ae60", lw=2, ls="--", label="Poisson")
        ax.set_xscale("log")
        ax.set_xlabel("window length $L$ (mean spacings)")
        ax.set_ylabel(ylabel)
        top = 1.35 * max(
            float(gue_curve[-1]),
            float(np.nanmax(emp_xp)),
            float(np.nanmax(emp_z)),
        )
        ax.set_ylim(0.0, top)
        ax.legend(loc="upper left", fontsize=8)

    ax_v.set_title(r"Number variance $\Sigma^2(L)$")
    ax_d.set_title(r"Dyson--Mehta $\Delta_3(L)$")

    fig.suptitle(
        title
        or "Deformed-$xp$ (picket fence) vs the Riemann zeros (GUE): "
        "mean density matched, fluctuations not"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def deformed_xp_eigh_convergence_figure(
    errors_by_n: dict[int, np.ndarray],
    *,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save the deformed-``xp`` GPU-eigh convergence figure (issue #31).

    One curve per Galerkin basis size ``N``: ``|eig_k - secular_k|`` (the GPU
    eigenvalue vs the #23 secular root) against the eigenvalue index ``k``, on a
    log ``y`` axis. The signature of spectral convergence is a machine-precision
    floor for the resolved low modes and a sharp cliff that moves to higher ``k``
    as ``N`` grows — the dense assembly reproducing the secular reference. Returns
    the path written.
    """
    sizes = sorted(errors_by_n)
    cmap = plt.get_cmap("viridis")
    span = (len(sizes) - 1) or 1

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, n in enumerate(sizes):
        err = np.asarray(errors_by_n[n], dtype=np.float64)
        k = np.arange(1, err.size + 1)
        ax.semilogy(
            k,
            np.maximum(err, 1e-16),  # floor for the log axis
            color=cmap(0.12 + 0.78 * i / span),
            lw=1.4,
            marker="o",
            ms=3,
            label=f"$N={n}$",
        )
    ax.set_xlabel("eigenvalue index $k$")
    ax.set_ylabel(r"$|\,\mathrm{eig}_k - E_k^{\mathrm{secular}}\,|$")
    ax.set_title(title or "Deformed-$xp$ GPU eigensolve vs the secular reference")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title="basis size", loc="upper left")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def dirichlet_locator_figure(
    grid: np.ndarray,
    abs_m: np.ndarray,
    peaks: np.ndarray,
    true_zeros: np.ndarray,
    *,
    out_path: Path | str,
    n: int | None = None,
    modulus: int | None = None,
    index: int | None = None,
    height: float | None = None,
    label: str | None = None,
    title: str | None = None,
) -> Path:
    """Save the Dirichlet-``L`` mirror locator scan ``|M'_z(E)|`` (issue #60).

    The teaching figure for the prime-driven locator (#25/#42). Plots the forward
    scan ``|M'_z(E)|`` (the truncated partial sum of ``1/L(s, chi)``) against the
    ordinate ``E``; the located peaks — the forward output, found from the character
    alone — are marked, and the **independently computed** zeros of ``L(s, chi)``
    are drawn as dashed vertical comparison lines. A peak landing on a line is the
    locator pulling that zero out of the primes; the zeros are shown only to score
    the peaks, never fed in (forward, not inverse). Pass ``height`` to draw the
    peak-detection threshold as a dotted guide. ``n`` / ``modulus`` / ``index``
    only label the title. Returns the path written.
    """
    grid = np.asarray(grid, dtype=np.float64)
    abs_m = np.asarray(abs_m, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.float64)
    true_zeros = np.asarray(true_zeros, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(
        grid,
        abs_m,
        color="#5b9bd5",
        lw=1.3,
        label=label or r"locator $|M'_z(E)|$",
    )
    # Independent L-zeros: dashed vertical comparison lines (scoring only).
    for i, z in enumerate(true_zeros):
        ax.axvline(
            z,
            color="#c0392b",
            lw=1.0,
            ls="--",
            alpha=0.7,
            label=r"independent $L(s,\chi)$ zeros" if i == 0 else None,
        )
    # Located peaks: the forward output, marked at their scan height.
    if peaks.size:
        peak_h = np.interp(peaks, grid, abs_m)
        ax.plot(
            peaks,
            peak_h,
            ls="none",
            marker="v",
            ms=8,
            color="#27ae60",
            label="located peaks (forward)",
        )
    if height is not None:
        ax.axhline(
            height,
            color="#7f7f7f",
            lw=1.0,
            ls=":",
            alpha=0.7,
            label="peak threshold",
        )
    ax.set_xlabel(r"ordinate $E$  (critical line $s = 1/2 + iE$)")
    ax.set_ylabel(r"$|M'_z(E)|$")
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_ylim(bottom=0.0)
    if title is None:
        if modulus is not None:
            idx = "" if index is None else f", index {index}"
            nstr = "" if n is None else f", $n={n:,}$ mirrors"
            title = rf"Dirichlet-$L$ mirror locator: $\chi$ mod {modulus}{idx}{nstr}"
        else:
            title = "Dirichlet-$L$ mirror locator"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
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


def katz_sarnak_density_figure(
    centres: np.ndarray,
    density: np.ndarray,
    *,
    out_path: Path | str,
    n_members: int | None = None,
    x_disc: float = 1.0,
    title: str | None = None,
) -> Path:
    """Save the family 1-level density vs the symmetry-type kernels (issue #51).

    The teaching figure for Katz-Sarnak universality. Bars are the empirical
    1-level density ``W(x)`` of the quadratic Dirichlet family (pooled low-lying
    zeros rescaled by conductor, :func:`katz_sarnak.family_one_level_density`); the
    three curves are the parameter-free random-matrix kernels — **symplectic**
    ``1 - sin(2 pi x)/(2 pi x)`` (the prediction, suppressed at the centre), unitary
    ``1`` (flat), and even-orthogonal ``1 + sin(2 pi x)/(2 pi x)`` (enhanced). The
    discrimination window ``x <= x_disc`` (where the kernels split) is shaded, and
    the closest kernel — the forward verdict — is annotated. Forward: the zeros are
    produced per member and only compared; nothing is fit. Returns the path written.
    """
    centres = np.asarray(centres, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    width = float(centres[1] - centres[0]) if centres.size > 1 else 0.25
    grid = np.linspace(1e-3, float(centres[-1] + 0.5 * width), 400)
    verdict = katz_sarnak.classify_symmetry(centres, density, x_disc=x_disc)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    members = "" if n_members is None else f" ({n_members} L-functions)"
    ax.bar(
        centres,
        density,
        width=width,
        align="center",
        color="#cfe3f7",
        edgecolor="#5b9bd5",
        label="quadratic family $W(x)$" + members,
    )
    ax.axvspan(
        0.0, x_disc, color="#f6d6d0", alpha=0.35, label=rf"$x\leq{x_disc:g}$ window"
    )
    ax.plot(
        grid,
        katz_sarnak.symplectic_density(grid),
        color="#c0392b",
        lw=2,
        label=r"symplectic $1-\frac{\sin 2\pi x}{2\pi x}$",
    )
    ax.plot(
        grid,
        katz_sarnak.unitary_density(grid),
        color="#27ae60",
        lw=2,
        ls="--",
        label="unitary $1$",
    )
    ax.plot(
        grid,
        katz_sarnak.orthogonal_even_density(grid),
        color="#8e44ad",
        lw=2,
        ls="-.",
        label=r"orthogonal $1+\frac{\sin 2\pi x}{2\pi x}$",
    )
    ax.set_xlabel(r"rescaled height $x = \gamma\,\log q / 2\pi$")
    ax.set_ylabel("1-level density $W(x)$")
    ax.set_xlim(0.0, float(grid[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        title
        or f"Katz--Sarnak: quadratic Dirichlet family is {verdict} (suppressed at $x=0$)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# CCM convergence-law figures (issue #65)
# ----------------------------------------------------------------------------


def ccm_convergence_artifact_figure(study: dict, *, out_path: Path | str) -> Path:
    """Genuine (mpmath) vs fp64 low-zero error across the cutoff (the #65 headline).

    ``study`` is the dict from ``run_ccm_convergence.study_artifact``. The genuine
    error collapses **super-exponentially** in the cutoff ``x`` (Groskin), while the
    fp64 error stays ``O(1..10)`` — it is ``xi``-corruption from the precision wall,
    not finite-cutoff error. So a fp64 ("~7-digit") inverse-log "measurement" is
    largely measuring the wall.
    """
    rows = study["rows"]
    xs = [r["x"] for r in rows]
    genuine = [max(r["genuine"], 1e-300) for r in rows]
    fp64 = [max(r["fp64"], 1e-300) for r in rows]
    corruption = [max(r["corruption"], 1e-300) for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(
        xs,
        genuine,
        color="#c0392b",
        lw=2,
        marker="o",
        label="genuine error (mpmath): super-exponential",
    )
    ax.semilogy(
        xs, fp64, color="#5b9bd5", lw=2, marker="s", label="fp64 error: wall-limited"
    )
    ax.semilogy(
        xs,
        corruption,
        color="#7f8c8d",
        lw=1.4,
        ls="--",
        marker="^",
        label=r"$|\nu^{\mathrm{fp64}}-\nu^{\mathrm{mpmath}}|$ (corruption)",
    )
    ax.set_xlabel(r"prime cutoff $x=\lambda^2$")
    ax.set_ylabel(f"max error over first {study['low']} zeros")
    ax.set_title(f"CCM low-zero convergence: genuine vs fp64 ($N={study['N']}$)")
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_convergence_edge_figure(study: dict, *, out_path: Path | str) -> Path:
    """Per-index error profile vs the Heisenberg floor ``1/(4 ln lambda)`` (Thm 3.1).

    ``study`` is the dict from ``run_ccm_convergence.study_edge``. The low zeros sit
    far below the floor (super-exponentially small); the per-index error climbs
    toward the floor only at the resolution edge ``k -> N`` — so the inverse-log
    floor is an edge phenomenon, not a property of the tracked spectrum.
    """
    errs = np.asarray(study["errors"], dtype=np.float64)
    errs = np.clip(errs, 1e-300, None)
    k = np.arange(1, errs.size + 1)
    bound = study["bound"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(
        k, errs, color="#5b9bd5", lw=1.4, marker="o", ms=3, label=r"$|\nu_k-\zeta_k|$"
    )
    ax.axhline(
        bound,
        color="#c0392b",
        lw=2,
        ls="--",
        label=r"Heisenberg floor $1/(4\ln\lambda)$",
    )
    if study.get("k_cross"):
        ax.axvline(
            study["k_cross"],
            color="#7f8c8d",
            lw=1.2,
            ls=":",
            label=f"reaches floor (edge) at $k={study['k_cross']}$",
        )
    ax.set_xlabel("zero index $k$")
    ax.set_ylabel(r"per-index error $|\nu_k-\zeta_k|$")
    ax.set_title(
        f"CCM error profile vs Heisenberg floor ($N={study['N']}$, $x={study['x']}$)"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
