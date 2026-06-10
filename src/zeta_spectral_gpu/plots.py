"""Matplotlib figures for the warm-up statistics.

Headless (Agg backend) so figures render in CI and on a display-less server. The
figures only ever *characterise* the zeros against the random-matrix surmises —
forward, not inverse.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from . import katz_sarnak, lehmer_census, li_criterion, spacing, zeros

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


def ccm_tracking_range_figure(study: dict, *, out_path: Path | str) -> Path:
    """The zero-tracking range ``k*(x)`` vs the prime cutoff (issue #53).

    ``study`` is the dict from ``run_ccm_convergence.study_tracking``. Left: the
    measured tracked-block length ``k*(x)`` (filled = interior, hollow = capped at
    the truncation ``N``), with the self-consistent linear-law overlay
    ``#{zeros < c_hat * x}`` and the ``N`` ceiling. Right: the edge ordinate ratio
    ``t*(x)/x``; if the tracking height grows linearly in the cutoff it sits on a
    plateau (the measured constant ``c_hat``, shown against ``2 pi`` and ``2 pi e``).
    This is the quantitative form of the #18 observation that the GUE-tracking range
    extends with the prime cutoff. Forward: zeros score only.
    """
    rows = sorted(study["rows"], key=lambda r: r["x"])
    xs = np.array([r["x"] for r in rows], dtype=np.float64)
    kstar = np.array([r["k_star"] for r in rows], dtype=np.float64)
    capped = np.array([bool(r["capped"]) for r in rows])
    ratio = np.array([r["ratio"] if r["ratio"] is not None else np.nan for r in rows])
    N = study["N"]
    c_hat = study.get("c_hat", float("nan"))
    interior = ~capped

    fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Left: k*(x) with the linear-law overlay and the N ceiling.
    if study["rows"][0].get("k_pred") is not None:
        kpred = np.array([r["k_pred"] for r in rows], dtype=np.float64)
        ax_k.plot(
            xs,
            kpred,
            color="#c0392b",
            lw=1.8,
            ls="--",
            label=rf"$\#\{{\zeta_k < \hat c\,x\}}$, $\hat c={c_hat:.1f}$",
        )
    ax_k.axhline(N, color="#7f8c8d", lw=1.0, ls=":", label=f"truncation $N={N}$")
    if interior.any():
        ax_k.plot(
            xs[interior],
            kstar[interior],
            color="#5b9bd5",
            marker="o",
            ms=6,
            lw=1.6,
            label=r"measured $k^*(x)$",
        )
    if capped.any():
        ax_k.plot(
            xs[capped],
            kstar[capped],
            color="#5b9bd5",
            marker="o",
            ms=6,
            lw=0,
            mfc="white",
            label=r"$k^*$ capped at $N$",
        )
    ax_k.set_xlabel(r"prime cutoff $x = \lambda^2$")
    ax_k.set_ylabel(r"zero-tracking range $k^*(x)$")
    ax_k.set_title("How far up the spectrum tracks the zeros")
    ax_k.legend(fontsize=8, loc="upper left")

    # Right: the t*/x plateau (the linear-height law).
    ax_r.axhline(
        2 * np.pi * np.e, color="#7f7f7f", lw=1.0, ls=":", label=r"$2\pi e=17.08$"
    )
    ax_r.axhline(2 * np.pi, color="#bdc3c7", lw=1.0, ls=":", label=r"$2\pi=6.28$")
    if np.isfinite(c_hat):
        ax_r.axhline(
            c_hat, color="#c0392b", lw=2, ls="--", label=rf"$\hat c={c_hat:.2f}$"
        )
    if interior.any():
        ax_r.plot(
            xs[interior],
            ratio[interior],
            color="#27ae60",
            marker="s",
            ms=6,
            lw=1.6,
            label=r"interior $t^*/x$",
        )
    if capped.any():
        ax_r.plot(
            xs[capped],
            ratio[capped],
            color="#27ae60",
            marker="s",
            ms=6,
            lw=0,
            mfc="white",
            label=r"capped (understates $t^*$)",
        )
    ax_r.set_xlabel(r"prime cutoff $x = \lambda^2$")
    ax_r.set_ylabel(r"tracking-height ratio $t^*(x)\,/\,x$")
    ax_r.set_title(r"Is the tracking height $t^*=\zeta_{k^*}$ linear in $x$?")
    ax_r.set_ylim(bottom=0.0)
    ax_r.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        rf"CCM operator: the zero-tracking range grows with the prime cutoff "
        rf"($N={N}$, rel.\ tol.\ ${study['rel_tol']:g}$)"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_edge_corruption_figure(study: dict, *, out_path: Path | str) -> Path:
    """Where does fp64 ``xi``-corruption land — low band or the edge? (issue #82).

    ``study`` is the dict from ``run_ccm_convergence.study_edge_corruption``. Left:
    the per-index profile at the deepest truncation — the genuine error climbs
    super-exponentially from the low zeros toward the resolution edge, the
    ``xi``-corruption is **large in the low / near-null band and decays toward the
    edge** (the edge eigenvalues are pinned to the bulk poles ``d_n`` and robust),
    so the fp64 error tracks the corruption in the low band and the genuine error at
    the edge. Right: the crossover — the low-band corruption is bounded in ``N`` while
    the genuine edge error grows ``~ zeta_N``; above the crossover ``N`` fp64's
    uniform error ``E`` is set by the genuine (robust) edge, so a fp64 Conj-4.1 sweep
    is edge-dominated and plausibly genuine. Forward: zeros score only.
    """
    focus = study["focus"]
    rows = sorted(study["rows"], key=lambda r: r["N"])
    clip = 1e-300

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Left: per-index profile at the deepest N.
    k = np.arange(1, focus["count"] + 1)
    genuine = np.clip(np.asarray(focus["genuine"], dtype=np.float64), clip, None)
    fp64e = np.clip(np.asarray(focus["fp64_error"], dtype=np.float64), clip, None)
    corr = np.clip(np.asarray(focus["corruption"], dtype=np.float64), clip, None)
    ax_p.semilogy(
        k,
        genuine,
        color="#c0392b",
        lw=1.6,
        marker="o",
        ms=3,
        label=r"genuine $|\nu_k^{\mathrm{mpmath}}-\zeta_k|$",
    )
    ax_p.semilogy(
        k,
        fp64e,
        color="#5b9bd5",
        lw=1.4,
        marker="s",
        ms=3,
        label=r"fp64 $|\nu_k^{\mathrm{fp64}}-\zeta_k|$",
    )
    ax_p.semilogy(
        k,
        corr,
        color="#7f8c8d",
        lw=1.4,
        ls="--",
        marker="^",
        ms=3,
        label=r"corruption $|\nu_k^{\mathrm{fp64}}-\nu_k^{\mathrm{mpmath}}|$",
    )
    if focus.get("k_floor"):
        ax_p.axvline(
            focus["k_floor"] + 1,
            color="#27ae60",
            lw=1.2,
            ls=":",
            label=f"resolution edge $k={focus['k_floor']}$",
        )
    ax_p.set_xlabel("eigenvalue index $k$")
    ax_p.set_ylabel("error")
    ax_p.set_title(
        f"Per-index profile ($N={focus['N']}$, $x={study['x']}$): "
        "corruption is low-band"
    )
    ax_p.legend(fontsize=8, loc="lower right")

    # Right: the crossover in N.
    ns = np.array([r["N"] for r in rows], dtype=np.float64)
    lo_corr = np.array([r["low_corruption"] for r in rows])
    ed_gen = np.array([r["edge_genuine"] for r in rows])
    e_fp64 = np.array([r["E_fp64"] for r in rows])
    ax_c.plot(
        ns,
        lo_corr,
        color="#7f8c8d",
        lw=1.8,
        marker="^",
        label="low-band corruption (bounded)",
    )
    ax_c.plot(
        ns,
        ed_gen,
        color="#c0392b",
        lw=1.8,
        marker="o",
        label=r"genuine edge error $\sim\zeta_N$",
    )
    ax_c.plot(
        ns,
        e_fp64,
        color="#5b9bd5",
        lw=1.4,
        ls="--",
        marker="s",
        label=r"fp64 uniform error $E$",
    )
    if study.get("crossover_N"):
        ax_c.axvline(
            study["crossover_N"],
            color="#27ae60",
            lw=1.2,
            ls=":",
            label=f"crossover $N={study['crossover_N']}$",
        )
    ax_c.set_xlabel("truncation $N$")
    ax_c.set_ylabel("error magnitude")
    ax_c.set_title("Crossover: genuine edge overtakes low-band corruption")
    ax_c.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        r"fp64 $\xi$-corruption is confined to the low band; the edge is "
        "pole-pinned and robust (#82)"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def li_coefficients_figure(result, *, out_path: Path | str) -> Path:
    """Forward Li coefficients ``lambda_n`` and the RH growth law (issue #52).

    ``result`` is a ``li_criterion.LiCriterionResult``. Top panel: the computed
    ``lambda_n`` (forward, from ``log xi``) as bars, coloured by sign -- RH says
    none ever dip below the zero baseline -- with the asymptotic main term
    ``(n/2)(log n + gamma - 1 - log 2pi)`` overlaid where it is positive. Bottom
    panel: the relative deviation ``|lambda_n - main(n)| / lambda_n`` on a log axis,
    shrinking as ``n`` grows -- the growth law asserting itself.
    """
    n = np.arange(1, result.n_max + 1)
    lam = np.array([float(c) for c in result.coefficients], dtype=np.float64)
    main = np.array([float(li_criterion.li_main_term(int(k))) for k in n])

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.0, 6.2), sharex=True, height_ratios=[2, 1]
    )

    pos = lam >= 0
    ax0.bar(
        n[pos],
        lam[pos],
        width=0.8,
        color="#cfe3f7",
        edgecolor="#5b9bd5",
        label=r"$\lambda_n \geq 0$ (RH-consistent)",
    )
    if (~pos).any():
        ax0.bar(
            n[~pos],
            lam[~pos],
            width=0.8,
            color="#f5c6c6",
            edgecolor="#c0392b",
            label=r"$\lambda_n < 0$ (would refute RH)",
        )
    main_pos = main > 0
    ax0.plot(
        n[main_pos],
        main[main_pos],
        color="#c0392b",
        lw=1.8,
        ls="--",
        label=r"$\frac{n}{2}(\ln n + \gamma - 1 - \ln 2\pi)$",
    )
    ax0.axhline(0.0, color="#7f8c8d", lw=1.0)
    ax0.set_ylabel(r"Li coefficient $\lambda_n$")
    ax0.set_title(
        f"Forward Li criterion: $\\lambda_n$ from $\\log\\xi$ "
        f"($n\\leq{result.n_max}$, dps$={result.dps}$; "
        f"min $\\lambda={float(result.min_value):.3g}$ at $n={result.min_index}$)"
    )
    ax0.legend(loc="upper left", fontsize=9)

    rel = np.array([float(r) for r in result.main_term_relative_error()])  # n = 2..
    ax1.semilogy(n[1:], rel, color="#5b9bd5", lw=1.4, marker="o", ms=3)
    ax1.set_xlabel("index $n$")
    ax1.set_ylabel(r"$|\lambda_n - \mathrm{main}(n)| / \lambda_n$")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def li_family_figure(result, *, out_path: Path | str) -> Path:
    """Forward GRH Li margins over a Dirichlet family (issue #71).

    ``result`` is a ``li_criterion_family.FamilyLiResult``. Each family member is one
    point: its GRH margin ``min_n Re lambda_n(chi)`` -- the closest any computed Li
    coefficient comes to the ``Re lambda_n = 0`` boundary GRH forbids it to cross.
    Real (quadratic) and complex characters are drawn separately (the symmetry-type
    contrast). Every point sitting above the dashed baseline is the forward,
    family-wide GRH-consistency verdict; a point dipping below would refute GRH for
    that character.
    """
    members = result.members
    x = np.arange(len(members))
    min_re = np.array([float(m.min_re) for m in members])
    is_real = np.array([m.is_real for m in members])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.scatter(
        x[is_real],
        min_re[is_real],
        s=26,
        color="#5b9bd5",
        edgecolor="#2f6aa8",
        label=r"real (quadratic) $\chi$",
        zorder=3,
    )
    if (~is_real).any():
        ax.scatter(
            x[~is_real],
            min_re[~is_real],
            s=30,
            color="#c0392b",
            marker="^",
            edgecolor="#7d1f14",
            label=r"complex $\chi$",
            zorder=3,
        )
    ax.axhline(
        0.0,
        color="#c0392b",
        lw=1.2,
        ls="--",
        label=r"GRH boundary ($\mathrm{Re}\,\lambda_n = 0$)",
    )
    # Linear scale with the 0 boundary in view: it shows how close the tightest
    # margin comes to the line GRH forbids any coefficient to cross.
    hi = float(np.max(min_re)) if min_re.size else 1.0
    ax.set_ylim(bottom=min(0.0, float(np.min(min_re))) - 0.05 * hi, top=1.1 * hi)
    worst = result.worst_member
    ax.set_xlabel("family member (enumeration order)")
    ax.set_ylabel(r"GRH margin  $\min_n \mathrm{Re}\,\lambda_n(\chi)$")
    ax.set_title(
        f"Forward GRH Li criterion over the {result.kind} family "
        f"({result.n_members} characters, $n\\leq{result.n_max}$)\n"
        f"tightest margin {float(worst.min_re):.3g} at {worst.label}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ccm_intermediate_stats_figure(study: dict, *, out_path: Path | str) -> Path:
    """The Seba / rank-one read of the CCM pole-locked tail (issue #87).

    ``study`` is the dict from ``run_ccm_convergence.study_intermediate``. Four
    panels. (a) The effective-coupling profile ``w_n = |xi_n / R_n| / Delta`` vs
    the pole ordinate: it fluctuates scale-free at O(0.1..1) with no break at the
    measured tracking height ``t*`` (dotted) — the intermediate regime, and why
    ``t*`` is invisible to coupling magnitude. (b) Windowed ``<r~>`` vs window
    center: the measured curves against the local two-pole prediction (dashed)
    and the Poisson / semi-Poisson / GUE / picket references. (c) The pooled tail
    spacing histogram in units of the pole spacing, measured vs predicted.
    (d) The two boundaries vs ``x``: the predicted-occupancy deficit plateau
    (the density crossover, tracking ``2 pi x``) and the measured ``t*(x)`` (the
    #53 law ``~ 11.75 x``) — the local read sees the former, not the latter.
    Forward: zeros score only.
    """
    rows = sorted(study["rows"], key=lambda r: r["x"])
    cmap = plt.get_cmap("viridis")
    colors = {r["x"]: cmap(i / max(1, len(rows) - 1)) for i, r in enumerate(rows)}

    fig, ((ax_w, ax_r), (ax_h, ax_t)) = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # (a) coupling profile w_n vs pole ordinate, with the measured t* marks.
    for r in rows:
        poles = np.asarray(r["poles"], dtype=np.float64)
        w = np.clip(np.asarray(r["w"], dtype=np.float64), 1e-300, None)
        ax_w.semilogy(
            poles, w, color=colors[r["x"]], lw=1.0, alpha=0.75, label=f"$x={r['x']}$"
        )
        if r.get("t_star"):
            ax_w.axvline(r["t_star"], color=colors[r["x"]], lw=0.9, ls=":")
    ax_w.axhline(0.5, color="#c0392b", lw=1.2, ls="--", label=r"$w = 1/2$")
    ax_w.set_xlabel(r"pole ordinate $d_n = 2\pi n / L$")
    ax_w.set_ylabel(r"$w_n = |\xi_n / R_n| \, / \, \Delta$")
    ax_w.set_title(r"Effective coupling: scale-free, no break at $t^*$ (dotted)")
    ax_w.legend(fontsize=8, loc="lower right")

    # (b) windowed <r~>: measured (solid) vs the local two-pole model (dashed).
    for ref, name, c in (
        (1.0, "picket", "#7f8c8d"),
        (0.6027, "GUE", "#c0392b"),
        (0.5, "semi-Poisson", "#8e44ad"),
        (0.3863, "Poisson", "#bdc3c7"),
    ):
        ax_r.axhline(ref, color=c, lw=0.9, ls=":")
        ax_r.annotate(
            name,
            (0.995, ref),
            xycoords=("axes fraction", "data"),
            fontsize=7,
            color=c,
            ha="right",
            va="bottom",
        )
    for r in rows:
        meas = np.asarray(r["windowed_meas"], dtype=np.float64)
        if meas.size:
            ax_r.plot(
                meas[:, 0],
                meas[:, 1],
                color=colors[r["x"]],
                lw=1.4,
                label=f"$x={r['x']}$ measured",
            )
        pred = np.asarray(r["windowed_pred"], dtype=np.float64)
        if pred.size:
            ax_r.plot(pred[:, 0], pred[:, 1], color=colors[r["x"]], lw=1.4, ls="--")
    ax_r.set_xlabel("window-center ordinate")
    ax_r.set_ylabel(rf"windowed $\langle\tilde r\rangle$ (window {study['window']})")
    ax_r.set_title("Measured vs local two-pole theory (dashed)")
    ax_r.set_ylim(0.3, 1.05)
    ax_r.legend(fontsize=7, loc="lower right")

    # (c) pooled tail spacing histogram, in units of the pole spacing Delta.
    edges = np.asarray(study["hist_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    ax_h.bar(
        centers,
        study["hist_meas"],
        width=width,
        color="#cfe3f7",
        edgecolor="#5b9bd5",
        label="measured tail",
    )
    ax_h.step(
        edges,
        np.r_[study["hist_pred"], study["hist_pred"][-1]],
        where="post",
        color="#c0392b",
        lw=1.6,
        label="local two-pole theory",
    )
    ax_h.axvline(1.0, color="#7f8c8d", lw=0.9, ls=":", label=r"$s = \Delta$ (picket)")
    ax_h.set_xlabel(r"tail spacing $s / \Delta$")
    ax_h.set_ylabel("density")
    ax_h.set_title(r"Tail spacings above $t^*$, pooled over cutoffs")
    ax_h.legend(fontsize=8, loc="upper left")

    # (d) the two boundaries: the deficit plateau (density crossover) vs t*.
    xs = np.array([r["x"] for r in rows], dtype=np.float64)
    t_dens = np.array(
        [r["t_dens"] if r["t_dens"] is not None else np.nan for r in rows]
    )
    lo = np.array(
        [r["t_dens_first"] if r["t_dens_first"] is not None else np.nan for r in rows]
    )
    hi = np.array(
        [r["t_dens_last"] if r["t_dens_last"] is not None else np.nan for r in rows]
    )
    t_star = np.array(
        [r["t_star"] if r.get("t_star") else np.nan for r in rows], dtype=np.float64
    )
    ax_t.errorbar(
        xs,
        t_dens,
        yerr=np.vstack([t_dens - lo, hi - t_dens]),
        color="#5b9bd5",
        marker="o",
        ms=6,
        lw=1.6,
        capsize=3,
        label="deficit plateau (predicted occupancy)",
    )
    ax_t.plot(
        xs,
        t_star,
        color="#27ae60",
        marker="s",
        ms=6,
        lw=1.6,
        label=r"measured $t^*(x) = \zeta_{k^*}$ (#53)",
    )
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax_t.plot(
        grid,
        2 * np.pi * grid,
        color="#5b9bd5",
        lw=1.2,
        ls="--",
        label=r"density crossover $t = 2\pi x$",
    )
    ax_t.plot(
        grid,
        11.75 * grid,
        color="#c0392b",
        lw=1.2,
        ls="--",
        label=r"#53 law $t = 11.75\,x$",
    )
    ax_t.set_xlabel(r"prime cutoff $x = \lambda^2$")
    ax_t.set_ylabel("ordinate")
    ax_t.set_title(r"The local read sees the density crossover, not $t^*$")
    ax_t.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        rf"CCM pole-locked tail as a rank-one (Seba) point process ($N={study['N']}$)"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def davenport_heilbronn_growth_figure(study: dict, *, out_path: Path | str) -> Path:
    """The #85 growth dichotomy on a genuine off-line zero (log-log profiles).

    ``study``: ``truncations`` (1d int array), ``profiles`` — an ordered mapping
    ``label -> |M(n)|`` array — ``slopes`` (``label -> measured slope``), and
    ``predicted`` (the off-line exponent ``sigma_c - 1/2``). The off-line curve
    should rise visibly above the no-Euler-product background of the other two.
    """
    n = np.asarray(study["truncations"], dtype=np.float64)
    colors = {"off-line": "#c0392b", "on-line": "#2c6fbb", "generic": "#27ae60"}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, profile in study["profiles"].items():
        slope = study["slopes"][label]
        color = colors.get(label.split()[0], None)
        ax.loglog(
            n,
            np.asarray(profile, dtype=np.float64),
            lw=1.2,
            color=color,
            label=f"{label} (slope {slope:+.3f})",
        )
    anchor = study["profiles"]["off-line"][-1] if "off-line" in study["profiles"] else 1
    guide = anchor * (n / n[-1]) ** study["predicted"]
    ax.loglog(
        n,
        guide,
        ls="--",
        color="#666666",
        lw=1.0,
        label=rf"predicted $n^{{{study['predicted']:.3f}}}$",
    )
    ax.set_xlabel("truncation $n$")
    ax.set_ylabel(r"$|\sum_{k \leq n} c(k)\, k^{-1/2 - iE}|$")
    ax.set_title("Davenport–Heilbronn: growth law at a genuine off-line zero")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def davenport_heilbronn_stats_figure(study: dict, *, out_path: Path | str) -> Path:
    """The #85 statistics control: f keeps repulsion; the superposition does not.

    ``study``: ``spacings_f`` / ``spacings_union`` (unit-mean unfolded nearest-
    neighbour spacings of f's on-line zeros and of the two-L superposition),
    ``rtilde`` (``label -> mean folded ratio``), ``deficit`` (smooth-count
    deficit of f), ``t_max``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    grid = np.linspace(1e-3, 3.5, 400)
    panels = (
        ("f (Davenport–Heilbronn)", study["spacings_f"], "#cfe3f7", "#5b9bd5"),
        (
            r"superposition: $L_\chi \cup L_{\bar\chi}$",
            study["spacings_union"],
            "#fde2cf",
            "#e67e22",
        ),
    )
    for ax, (label, spac, fill, edge) in zip(axes, panels):
        s = np.asarray(spac, dtype=np.float64)
        centres, density = spacing.spacing_density(s, n_bins=40, s_max=3.5)
        width = float(centres[1] - centres[0])
        ax.bar(
            centres,
            density,
            width=width,
            align="center",
            color=fill,
            edgecolor=edge,
            label=f"{label} (N={s.size:,})",
        )
        ax.plot(
            grid, spacing.gue_wigner_surmise(grid), color="#c0392b", lw=1.6, label="GUE"
        )
        ax.plot(
            grid,
            spacing.poisson_surmise(grid),
            color="#27ae60",
            lw=1.6,
            ls="--",
            label="Poisson",
        )
        ax.set_xlabel("normalised spacing $s$")
        ax.set_xlim(0.0, 3.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("$p(s)$")
    ratios = ", ".join(
        rf"$\langle\tilde r\rangle_{{{k}}}={v:.3f}$" for k, v in study["rtilde"].items()
    )
    fig.suptitle(
        f"Davenport–Heilbronn statistics control to $t={study['t_max']:g}$: "
        f"{ratios}; on-line deficit {study['deficit']:.1f}"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def davenport_heilbronn_locator_figure(study: dict, *, out_path: Path | str) -> Path:
    """The #85 locator control: clean peaks for a genuine L, mounds for f.

    ``study``: ``grid``, ``abs_f`` / ``abs_chi`` (locator magnitudes with the
    ``1/f`` inverse weights and the genuine ``chi mu`` weights), ``threshold``,
    ``true_f`` / ``true_chi`` (independent on-line zeros), ``off_line``
    (off-line ordinates). The off-line ordinates should sit under the spurious
    mounds of the f trace and nowhere on the chi trace.
    """
    grid = np.asarray(study["grid"], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
    traces = (
        ("$1/f$ inverse weights", study["abs_f"], study["true_f"], "#5b9bd5"),
        (
            r"$\chi\,\mu$ weights ($L_\chi$)",
            study["abs_chi"],
            study["true_chi"],
            "#e67e22",
        ),
    )
    for ax, (label, abs_m, true_zeros, color) in zip(axes, traces):
        ax.plot(grid, np.asarray(abs_m, dtype=np.float64), lw=0.7, color=color)
        ax.axhline(study["threshold"], color="#666666", lw=1.0, ls="--")
        ax.plot(
            np.asarray(true_zeros, dtype=np.float64),
            np.full(len(true_zeros), -0.5),
            "|",
            ms=8,
            color="#2c3e50",
            label="independent on-line zeros",
        )
        for t in study["off_line"]:
            ax.axvline(t, color="#c0392b", lw=1.0, ls=":")
        ax.set_ylabel(f"$|M(E)|$\n{label}")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title(
        "Davenport–Heilbronn locator control (dotted red: off-line zero ordinates)"
    )
    axes[1].set_xlabel("energy $E$")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def zero_form_factor_figure(
    u: np.ndarray,
    s2_emp: np.ndarray,
    s2_pred: np.ndarray,
    ramp: np.ndarray,
    prime_marks: list[tuple[float, str]],
    *,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save |S(u)|^2 of the zeros vs the explicit-formula prime prediction.

    ``s2_emp``/``s2_pred`` are the empirical and predicted ``|S(u)|^2`` on the
    grid ``u``; ``ramp`` is the smoothed GUE/diagonal background and
    ``prime_marks`` the ``(log p^m, label)`` positions where the arithmetic
    peaks must sit (issue #84). Log scale: the peaks stand orders of magnitude
    above the ramp below the Heisenberg frequency.
    """
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.semilogy(u, s2_emp, color="#5b9bd5", lw=0.9, label="zeros $|S(u)|^2$")
    ax.semilogy(
        u,
        s2_pred,
        color="#c0392b",
        lw=1.0,
        ls="--",
        label=r"primes: $|\frac{1}{2\pi}\sum_n \Lambda(n) n^{-1/2} W(\log n - u)|^2$",
    )
    ax.semilogy(
        u,
        ramp,
        color="#27ae60",
        lw=1.6,
        ls=":",
        label=r"GUE diagonal ramp $\frac{u}{2\pi}\int w^2$",
    )
    top = float(np.nanmax(s2_emp)) * 3.0
    for pos, label in prime_marks:
        if u[0] <= pos <= u[-1]:
            ax.axvline(pos, color="#999999", lw=0.5, zorder=0)
            ax.annotate(
                label,
                (pos, top),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
    ax.set_xlabel(r"frequency $u$  (peaks at $u = \log p^m$)")
    ax.set_ylabel(r"$|S(u)|^2$")
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.set_title(title or "Spectral form factor of the zeros: primes beyond GUE")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pair_correlation_deviation_figure(
    eps: np.ndarray,
    emp: np.ndarray,
    cs: np.ndarray,
    gue: np.ndarray,
    plateau: float,
    *,
    out_path: Path | str,
    title: str | None = None,
) -> Path:
    """Save the arithmetic deviation of the zeros' pair correlation from GUE.

    All inputs are pair densities per unit raw separation ``eps`` over the same
    height window: ``emp`` (histogram), ``cs`` (Conrey-Snaith Theorem 4.1) and
    ``gue`` (sine kernel only); ``plateau`` is the uncorrelated level used to
    normalise. Top panel: R2 itself; bottom: the deviation from GUE, where the
    arithmetic (Bogomolny-Keating) terms are the entire signal (issue #84).
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 6.5), sharex=True, height_ratios=[2, 1]
    )
    axes[0].plot(eps, emp / plateau, color="#5b9bd5", lw=0.9, label="zeros (histogram)")
    axes[0].plot(
        eps, cs / plateau, color="#c0392b", lw=1.2, ls="--", label="Conrey-Snaith"
    )
    axes[0].plot(
        eps, gue / plateau, color="#27ae60", lw=1.2, ls=":", label="GUE sine kernel"
    )
    axes[0].set_ylabel(r"$R_2(\epsilon)$ / plateau")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].axhline(0.0, color="#666666", lw=0.8)
    axes[1].plot(
        eps,
        (emp - gue) / plateau,
        color="#5b9bd5",
        lw=0.9,
        label="zeros $-$ GUE",
    )
    axes[1].plot(
        eps,
        (cs - gue) / plateau,
        color="#c0392b",
        lw=1.2,
        ls="--",
        label="Conrey-Snaith $-$ GUE (arithmetic terms)",
    )
    axes[1].set_xlabel(r"raw ordinate separation $\epsilon$")
    axes[1].set_ylabel("deviation / plateau")
    axes[1].legend(loc="lower right", fontsize=8)

    axes[0].set_title(
        title or "Pair correlation of the zeros: lower-order arithmetic terms"
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def lehmer_census_figure(
    study: dict, *, out_path: Path | str, title: str | None = None
) -> Path:
    """Two-panel #86 readout: small-gap tail vs GUE, and the CSV census plane.

    ``study["windows"]`` rows carry ``label``, ``s`` (normalized gaps),
    ``floor`` (the fp64 resolution floor), and ``rows`` (``PairRow`` lists).
    Left: empirical cumulative gap distribution per height window (log-log)
    against the exact sine-kernel cube law and the Wigner-surmise CDF, with
    each window's fp64 floor marked. Right: every censused pair on the
    ``(s, Delta^2 g)`` plane with the CSV ``4/5`` Lehmer bar.
    """
    windows = study["windows"]
    colors = ["#5b9bd5", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    s_lo = 8e-3
    grid = np.geomspace(s_lo, 1.0, 300)
    fine = np.linspace(0.0, 1.0, 4001)
    wigner_cdf = np.cumsum(spacing.gue_wigner_surmise(fine)) * (fine[1] - fine[0])
    for k, w in enumerate(windows):
        s = np.sort(np.asarray(w["s"]))
        ecdf = np.searchsorted(s, grid, side="right") / s.size
        ok = ecdf > 0
        axes[0].loglog(
            grid[ok],
            ecdf[ok],
            color=colors[k % len(colors)],
            lw=1.6,
            label=f"{w['label']} (n={s.size:,})",
        )
        axes[0].axvline(
            w["floor"], color=colors[k % len(colors)], ls=":", lw=1.0, alpha=0.7
        )
    axes[0].loglog(
        grid,
        lehmer_census.gue_small_gap_cdf(grid),
        color="black",
        lw=1.2,
        ls="--",
        label=r"GUE $(\pi^2/9)s^3$",
    )
    axes[0].loglog(
        grid,
        np.interp(grid, fine, wigner_cdf),
        color="gray",
        lw=1.0,
        ls="-.",
        label="Wigner surmise CDF",
    )
    axes[0].set_xlabel("normalized gap $s$")
    axes[0].set_ylabel(r"$P(\mathrm{gap} < s)$")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("small-gap tail vs GUE repulsion (dotted: fp64 floors)")

    for k, w in enumerate(windows):
        rows = w["rows"]
        if not rows:
            continue
        ss = np.array([r.s for r in rows])
        qq = np.array([r.delta2g for r in rows])
        n_lehmer = sum(1 for r in rows if r.lam is not None)
        axes[1].loglog(
            ss,
            qq,
            ".",
            ms=3,
            alpha=0.5,
            color=colors[k % len(colors)],
            label=f"{w['label']}: {n_lehmer} Lehmer pairs",
        )
    axes[1].axhline(
        lehmer_census.CSV_THRESHOLD, color="black", ls="--", lw=1.2, label=r"CSV $4/5$"
    )
    axes[1].set_xlabel("normalized gap $s$")
    axes[1].set_ylabel(r"$\Delta^2 g$  (Lehmer pair below the bar)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("CSV census plane")

    fig.suptitle(title or "Lehmer-pair / small-gap census (#86)")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
