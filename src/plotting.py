"""Plotting helpers for the Chapter 4 uncertainty-set analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .score_tools import ResidualLaw, folded_normal_pdf, integration_upper_bound, mixture_pdf


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "blue": "#0B3C5D",
    "rust": "#B03A2E",
    "teal": "#007C89",
    "purple": "#7B3294",
    "gray": "#6A6A6A",
    "light_gray": "#B5B5B5",
    "black": "#222222",
    "gold": "#C49A00",
}

STYLES = {
    "standard_cp": {
        "color": COLORS["blue"],
        "linewidth": 2.2,
        "linestyle": "-",
        "label": "Standard CP",
    },
    "matched_rv": {
        "color": COLORS["rust"],
        "linewidth": 2.2,
        "linestyle": (0, (5, 2)),
        "label": "matched TV-RV",
    },
    "fixed_target_rv": {
        "color": COLORS["teal"],
        "linewidth": 2.2,
        "linestyle": "-",
        "label": "fixed-target TV-RV",
    },
    "fixed_target_set": {
        "color": COLORS["teal"],
        "linewidth": 2.2,
        "linestyle": "-",
        "label": "fixed target set",
    },
    "max_observed_risk": {
        "color": COLORS["blue"],
        "linewidth": 2.0,
        "linestyle": (0, (5, 2)),
        "label": "max observed risk",
    },
    "matched_bound": {
        "color": COLORS["gray"],
        "linewidth": 1.9,
        "linestyle": "-",
        "label": r"matched-$\gamma$ bound",
    },
    "matched_moment": {
        "color": COLORS["gray"],
        "linewidth": 1.9,
        "linestyle": "-",
    },
    "calibration_quantile": {
        "color": COLORS["blue"],
        "linewidth": 2.0,
        "linestyle": (0, (5, 2)),
        "label": "Standard CP",
    },
    "tv_worst": {
        "color": COLORS["purple"],
        "linewidth": 1.9,
        "linestyle": (0, (4, 2)),
    },
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)

    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _pretty_label(name: str) -> str:
    label = name.replace("_", " ")
    label = label.replace("obs mean", "observed env")
    label = label.replace("obs cov", "observed env")
    label = label.replace("obs mix", "observed env")
    return label


def _finite_xy(df: pd.DataFrame, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    y = df[y_col].to_numpy(dtype=float)
    mask = np.isfinite(y)
    return df.loc[mask, "gamma"].to_numpy(dtype=float), y[mask]


def _best_gamma(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None

    y = df[column].to_numpy(dtype=float)
    mask = np.isfinite(y)

    if not np.any(mask):
        return None

    idx = df.loc[mask, column].astype(float).idxmin()
    return float(df.loc[idx, "gamma"])


def _first_finite_gamma(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None

    y = df[column].to_numpy(dtype=float)
    mask = np.isfinite(y)

    if not np.any(mask):
        return None

    return float(df.loc[mask, "gamma"].iloc[0])


def _closest_gamma(df: pd.DataFrame, target: float) -> float:
    gammas = np.sort(df["gamma"].unique())
    idx = int(np.argmin(np.abs(gammas - target)))
    return float(gammas[idx])


def _vline(
    ax: plt.Axes,
    x: float,
    label: str,
    y_pos: float = 0.98,
    x_offset: float = 0.0,
    ha: str = "right",
) -> None:
    """Vertical reference line with a readable rotated label."""

    ax.axvline(
        x,
        color=COLORS["black"],
        linewidth=1.0,
        linestyle=(0, (2, 3)),
        alpha=0.75,
        zorder=0,
    )
    ax.text(
        x + x_offset,
        y_pos,
        label,
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha=ha,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
    )


def _vline_left_label(ax: plt.Axes, x: float, label: str, y_pos: float = 0.96) -> None:
    """Vertical line for very small gamma values, with the label shifted into the panel."""

    _vline(ax, x, label, y_pos=y_pos, x_offset=0.12, ha="left")


def _hline(ax: plt.Axes, y: float, label: str, x_pos: float = 0.98) -> None:
    ax.axhline(
        y,
        color=COLORS["black"],
        linewidth=1.0,
        linestyle=(0, (1, 3)),
        alpha=0.75,
        zorder=0,
    )
    ax.text(
        x_pos,
        y,
        label,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def _shade_vacuous(ax: plt.Axes, summary: pd.DataFrame, alpha: float) -> None:
    if "rho_wc" not in summary.columns:
        return

    gammas = summary["gamma"].to_numpy(dtype=float)
    rho = summary["rho_wc"].to_numpy(dtype=float)
    bad = rho >= 2.0 * alpha - 1e-12

    if not np.any(bad):
        return

    ymin, ymax = ax.get_ylim()
    ax.fill_between(
        gammas,
        ymin,
        ymax,
        where=bad,
        color=COLORS["light_gray"],
        alpha=0.15,
        step="mid",
        linewidth=0,
    )
    ax.set_ylim(ymin, ymax)


def _mask_by_finite_width(
    summary: pd.DataFrame,
    y_col: str,
    width_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gamma and y-values, masking entries where the associated width is not finite."""

    x = summary["gamma"].to_numpy(dtype=float)
    y = summary[y_col].to_numpy(dtype=float).copy()
    width = summary[width_col].to_numpy(dtype=float)
    y[~np.isfinite(width)] = np.nan
    return x, y


def _shared_legend(
    fig: plt.Figure,
    axes: np.ndarray | list[plt.Axes],
    ncol: int = 3,
    y: float = 1.00,
) -> None:
    """Add one deduplicated legend above a two-panel figure."""

    handles = []
    labels = []

    for ax in np.ravel(axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_"):
                continue
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=min(ncol, len(handles)),
        frameon=False,
        columnspacing=1.35,
        handlelength=2.8,
    )


def _axis_legend_above(ax: plt.Axes, ncol: int = 2, y: float = 1.01) -> None:
    """Place a local legend above a single panel."""

    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=min(ncol, len(handles)),
        frameon=False,
        columnspacing=1.0,
        handlelength=2.5,
    )


# ---------------------------------------------------------------------
# Axis scaling helpers
# ---------------------------------------------------------------------


FOCUS_GAMMA_MAX = 8.0
MATCHED_GAMMA_MAX = 5.0
MIXED_GAMMA_MAX = 6.0
DENSITY_GAMMAS = [0.0, 5.0]


def _apply_gamma_focus(ax: plt.Axes, df: pd.DataFrame, xmax: float = FOCUS_GAMMA_MAX) -> None:
    """Show the gamma range that contains the main tradeoff."""

    max_available = float(np.nanmax(df["gamma"].to_numpy(dtype=float)))
    ax.set_xlim(0.0, min(xmax, max_available))


def _values_in_focus(df: pd.DataFrame, columns: list[str], xmax: float = FOCUS_GAMMA_MAX) -> np.ndarray:
    """Return finite y-values from selected columns in the displayed gamma range."""

    mask = df["gamma"].to_numpy(dtype=float) <= xmax + 1e-12
    vals: list[np.ndarray] = []

    for col in columns:
        if col not in df.columns:
            continue

        y = df.loc[mask, col].to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        if y.size:
            vals.append(y)

    if not vals:
        return np.array([])

    return np.concatenate(vals)


def _set_zoomed_ylim(
    ax: plt.Axes,
    values: np.ndarray,
    lower_floor: float = 0.0,
    pad_frac: float = 0.10,
) -> None:
    """Set a readable y-axis for finite values."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))

    if np.isclose(ymin, ymax):
        delta = max(0.05, abs(ymax) * 0.1)
        ax.set_ylim(max(lower_floor, ymin - delta), ymax + delta)
        return

    pad = pad_frac * (ymax - ymin)
    ax.set_ylim(max(lower_floor, ymin - pad), ymax + pad)


def _set_coverage_ylim(
    ax: plt.Axes,
    values: np.ndarray,
    alpha: float,
    lower_floor: float | None = None,
) -> None:
    """Zoom coverage plots while keeping nominal coverage visible."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    ymin = float(min(np.nanmin(values), 1.0 - alpha))
    ymax = float(max(np.nanmax(values), 1.0 - alpha))
    pad = max(0.015, 0.08 * (ymax - ymin))

    lower = max(0.0, ymin - pad)
    if lower_floor is not None:
        lower = max(lower_floor, lower)

    ax.set_ylim(lower, min(1.01, ymax + pad))


# ---------------------------------------------------------------------
# Structured TV-RV settings
# ---------------------------------------------------------------------


def plot_fixed_target_diagnostics(summary: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    """Coverage and widths against a fixed target uncertainty set."""

    if "coverage_std_fixed_target" not in summary.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    gamma_risk_target = _best_gamma(summary, "risk_fixed_target")
    gamma_width_target = _best_gamma(summary, "width_rv_fixed_target")

    # Coverage under fixed target set
    ax = axes[0]
    ax.plot(
        summary["gamma"],
        summary["coverage_std_fixed_target"],
        **STYLES["standard_cp"],
    )

    x_current, y_current = _mask_by_finite_width(
        summary,
        "coverage_current_rv_on_fixed_target",
        "width_rv",
    )
    ax.plot(
        x_current,
        y_current,
        **STYLES["matched_rv"],
    )

    x_target, y_target = _mask_by_finite_width(
        summary,
        "coverage_target_rv_fixed_target",
        "width_rv_fixed_target",
    )
    ax.plot(
        x_target,
        y_target,
        **STYLES["fixed_target_rv"],
    )

    _hline(ax, 1.0 - alpha, r"$1-\alpha$")

    if gamma_risk_target is not None:
        _vline(ax, gamma_risk_target, r"$\gamma^\star_{\mathrm{risk},\eta}$", y_pos=0.98)

    if (
        gamma_width_target is not None
        and gamma_risk_target is not None
        and abs(gamma_width_target - gamma_risk_target) > 0.20
    ):
        _vline(ax, gamma_width_target, r"$\gamma^\star_{\mathrm{width},\eta}$", y_pos=0.58)

    _apply_gamma_focus(ax, summary, xmax=FOCUS_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        [
            "coverage_std_fixed_target",
            "coverage_current_rv_on_fixed_target",
            "coverage_target_rv_fixed_target",
        ],
        xmax=FOCUS_GAMMA_MAX,
    )
    _set_coverage_ylim(ax, vals, alpha)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Worst-case coverage on fixed target set")
    ax.grid(True, alpha=0.22)

    # Widths under fixed target set
    ax = axes[1]
    x_std, y_std = _finite_xy(summary, "width_std")
    ax.plot(x_std, y_std, **STYLES["standard_cp"])

    x_current, y_current = _finite_xy(summary, "width_rv")
    ax.plot(x_current, y_current, **STYLES["matched_rv"])

    x_target, y_target = _finite_xy(summary, "width_rv_fixed_target")
    ax.plot(x_target, y_target, **STYLES["fixed_target_rv"])

    if gamma_risk_target is not None:
        _vline(ax, gamma_risk_target, r"$\gamma^\star_{\mathrm{risk},\eta}$", y_pos=0.98)

    if (
        gamma_width_target is not None
        and gamma_risk_target is not None
        and abs(gamma_width_target - gamma_risk_target) > 0.20
    ):
        _vline(ax, gamma_width_target, r"$\gamma^\star_{\mathrm{width},\eta}$", y_pos=0.58)

    _apply_gamma_focus(ax, summary, xmax=FOCUS_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["width_std", "width_rv", "width_rv_fixed_target"],
        xmax=FOCUS_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Interval width")
    ax.grid(True, alpha=0.22)

    _shared_legend(fig, axes, ncol=3, y=1.00)
    _save(fig, output_dir, "fixed_target_coverage_width", rect=(0.0, 0.0, 1.0, 0.91))


def plot_fixed_target_risk_width(summary: pd.DataFrame, output_dir: Path) -> None:
    """Fixed-target risk and fixed-target robust-validation width."""

    if "risk_fixed_target" not in summary.columns or "width_rv_fixed_target" not in summary.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    gamma_risk_target = _best_gamma(summary, "risk_fixed_target")
    gamma_width_target = _best_gamma(summary, "width_rv_fixed_target")

    # Fixed-target residual moment / risk
    ax = axes[0]
    ax.plot(
        summary["gamma"],
        summary["risk_fixed_target"],
        **STYLES["fixed_target_set"],
    )

    if "risk_obs_max" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["risk_obs_max"],
            **STYLES["max_observed_risk"],
        )

    if "risk_current_set" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["risk_current_set"],
            **STYLES["matched_bound"],
        )

    if gamma_risk_target is not None:
        _vline(ax, gamma_risk_target, r"$\gamma^\star_{\mathrm{risk},\eta}$", y_pos=0.98)

    if (
        gamma_width_target is not None
        and gamma_risk_target is not None
        and abs(gamma_width_target - gamma_risk_target) > 0.20
    ):
        _vline(ax, gamma_width_target, r"$\gamma^\star_{\mathrm{width},\eta}$", y_pos=0.58)

    _apply_gamma_focus(ax, summary, xmax=FOCUS_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["risk_fixed_target", "risk_obs_max", "risk_current_set"],
        xmax=FOCUS_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Residual second moment / risk")
    ax.grid(True, alpha=0.22)

    # Fixed-target RV width
    ax = axes[1]

    x_target, y_target = _finite_xy(summary, "width_rv_fixed_target")
    ax.plot(
        x_target,
        y_target,
        **STYLES["fixed_target_rv"],
    )

    if "width_std" in summary.columns:
        x_std, y_std = _finite_xy(summary, "width_std")
        ax.plot(
            x_std,
            y_std,
            **STYLES["standard_cp"],
        )

    if gamma_risk_target is not None:
        _vline(ax, gamma_risk_target, r"$\gamma^\star_{\mathrm{risk},\eta}$", y_pos=0.98)

    if (
        gamma_width_target is not None
        and gamma_risk_target is not None
        and abs(gamma_width_target - gamma_risk_target) > 0.20
    ):
        _vline(ax, gamma_width_target, r"$\gamma^\star_{\mathrm{width},\eta}$", y_pos=0.58)

    _apply_gamma_focus(ax, summary, xmax=FOCUS_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["width_rv_fixed_target", "width_std"],
        xmax=FOCUS_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Interval width")
    ax.grid(True, alpha=0.22)

    _axis_legend_above(axes[0], ncol=2, y=1.01)
    _axis_legend_above(axes[1], ncol=2, y=1.01)
    _save(fig, output_dir, "fixed_target_risk_width", rect=(0.0, 0.0, 1.0, 0.84))


def plot_matched_radius_width(summary: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    """Matched-gamma TV radius and interval widths."""

    if "rho_wc" not in summary.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    gamma_width = _best_gamma(summary, "width_rv")
    gamma_finite = _first_finite_gamma(summary, "width_rv")

    # Matched TV radius
    ax = axes[0]
    ax.plot(
        summary["gamma"],
        summary["rho_wc"],
        color=COLORS["black"],
        linewidth=2.2,
        linestyle="-",
    )
    _hline(ax, 2.0 * alpha, r"$2\alpha$")

    if gamma_finite is not None:
        _vline_left_label(ax, gamma_finite, r"$\gamma_{\mathrm{fin}}$", y_pos=0.95)

    if (
        gamma_width is not None
        and gamma_finite is not None
        and abs(gamma_width - gamma_finite) > 0.20
    ):
        _vline(ax, gamma_width, r"$\gamma^\star_{\mathrm{width}}$", y_pos=0.58)

    _apply_gamma_focus(ax, summary, xmax=MATCHED_GAMMA_MAX)

    vals = _values_in_focus(summary, ["rho_wc"], xmax=MATCHED_GAMMA_MAX)
    if vals.size:
        vals = np.concatenate([vals, np.array([2.0 * alpha])])
        _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.12)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Matched TV radius $\rho_{\mathrm{wc}}$")
    ax.grid(True, alpha=0.22)

    # Matched widths
    ax = axes[1]

    x_std, y_std = _finite_xy(summary, "width_std")
    ax.plot(
        x_std,
        y_std,
        **STYLES["standard_cp"],
    )

    x_rv, y_rv = _finite_xy(summary, "width_rv")
    ax.plot(
        x_rv,
        y_rv,
        **STYLES["matched_rv"],
    )

    _apply_gamma_focus(ax, summary, xmax=MATCHED_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["width_std", "width_rv"],
        xmax=MATCHED_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    _shade_vacuous(ax, summary, alpha)

    if gamma_finite is not None:
        _vline_left_label(ax, gamma_finite, r"$\gamma_{\mathrm{fin}}$", y_pos=0.95)

    if (
        gamma_width is not None
        and gamma_finite is not None
        and abs(gamma_width - gamma_finite) > 0.20
    ):
        _vline(ax, gamma_width, r"$\gamma^\star_{\mathrm{width}}$", y_pos=0.58)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Interval width")
    ax.grid(True, alpha=0.22)

    _shared_legend(fig, axes, ncol=2, y=1.00)
    _save(fig, output_dir, "matched_radius_width", rect=(0.0, 0.0, 1.0, 0.91))


def plot_structured_mechanism(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    parameter: str,
) -> None:
    """Mechanism plot: projected means or projected variances shrink."""

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    observed = details[details["kind"] == "observed"].copy()
    env_colors = [COLORS["blue"], COLORS["rust"], COLORS["teal"], COLORS["purple"], COLORS["gold"]]

    if parameter == "mean":
        for i, (name, group) in enumerate(observed.groupby("name", sort=False)):
            group = group.sort_values("gamma")
            axes[0].plot(
                group["gamma"],
                group["mu"],
                color=env_colors[i % len(env_colors)],
                linewidth=1.8,
                label=_pretty_label(name),
            )

        axes[0].set_ylabel("Observed residual mean")

        axes[1].plot(
            summary["gamma"],
            summary["m_max"],
            color=COLORS["black"],
            linewidth=2.1,
            linestyle="-",
            label=r"$m_{\max}(\gamma)$",
        )
        axes[1].plot(
            summary["gamma"],
            summary["worst_m_tv"],
            color=STYLES["tv_worst"]["color"],
            linestyle=STYLES["tv_worst"]["linestyle"],
            linewidth=STYLES["tv_worst"]["linewidth"],
            label="TV-maximizing mean",
        )
        axes[1].set_ylabel("Projected test mean")

    elif parameter == "cov":
        for i, (name, group) in enumerate(observed.groupby("name", sort=False)):
            group = group.sort_values("gamma")
            axes[0].plot(
                group["gamma"],
                group["sigma"],
                color=env_colors[i % len(env_colors)],
                linewidth=1.8,
                label=_pretty_label(name),
            )

        axes[0].set_ylabel("Observed residual standard deviation")

        axes[1].plot(
            summary["gamma"],
            summary["h_max"],
            color=COLORS["black"],
            linewidth=2.1,
            linestyle="-",
            label=r"$h_{\max}(\gamma)$",
        )
        axes[1].plot(
            summary["gamma"],
            summary["worst_h_tv"],
            color=STYLES["tv_worst"]["color"],
            linestyle=STYLES["tv_worst"]["linestyle"],
            linewidth=STYLES["tv_worst"]["linewidth"],
            label="TV-maximizing perturbation",
        )
        axes[1].set_ylabel("Projected covariance perturbation")

    else:
        raise ValueError("parameter must be 'mean' or 'cov'")

    for ax in axes:
        ax.set_xlabel(r"$\gamma$")
        ax.grid(True, alpha=0.22)
        _apply_gamma_focus(ax, summary, xmax=FOCUS_GAMMA_MAX)

    _axis_legend_above(axes[0], ncol=3, y=1.01)
    _axis_legend_above(axes[1], ncol=2, y=1.01)
    _save(fig, output_dir, "mechanism_alignment", rect=(0.0, 0.0, 1.0, 0.84))


def plot_score_densities(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    weights: np.ndarray,
) -> None:
    """Calibration mixture and worst-case test score density at selected gamma values."""

    selected = []
    for target in DENSITY_GAMMAS:
        gamma = _closest_gamma(summary, target)
        if gamma not in selected:
            selected.append(gamma)

    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(4.8 * len(selected), 3.9),
        squeeze=False,
    )
    axes = axes[0]

    for ax, gamma in zip(axes, selected):
        sub = details[np.isclose(details["gamma"], gamma)]
        obs = sub[sub["kind"] == "observed"]
        tv = sub[sub["kind"] == "tv_worst_test"]

        if obs.empty or tv.empty:
            continue

        cal_laws = [
            ResidualLaw(row["name"], float(row["mu"]), float(row["sigma"]))
            for _, row in obs.iterrows()
        ]

        tv_row = tv.iloc[0]
        tv_law = ResidualLaw(str(tv_row["name"]), float(tv_row["mu"]), float(tv_row["sigma"]))

        upper = integration_upper_bound([*cal_laws, tv_law], multiplier=7.5)
        x = np.linspace(0.0, upper, 900)

        ax.plot(
            x,
            mixture_pdf(x, cal_laws, weights),
            color=COLORS["black"],
            linewidth=2.0,
            linestyle="-",
            label="Calibration score law",
        )
        ax.plot(
            x,
            folded_normal_pdf(x, tv_law.mu, tv_law.sigma),
            color=STYLES["tv_worst"]["color"],
            linewidth=STYLES["tv_worst"]["linewidth"],
            linestyle=STYLES["tv_worst"]["linestyle"],
            label="Worst-case test score law",
        )

        ax.set_title(rf"$\gamma={gamma:g}$")
        ax.set_xlabel("score")
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel("density")

    _shared_legend(fig, axes, ncol=2, y=1.00)
    _save(fig, output_dir, "score_densities_selected_gammas", rect=(0.0, 0.0, 1.0, 0.91))


def make_structured_plots(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    alpha: float,
    weights: np.ndarray,
    parameter: str,
) -> None:
    plot_fixed_target_diagnostics(summary, output_dir, alpha)
    plot_fixed_target_risk_width(summary, output_dir)
    plot_matched_radius_width(summary, output_dir, alpha)
    plot_structured_mechanism(summary, details, output_dir, parameter)
    plot_score_densities(summary, details, output_dir, weights)


# ---------------------------------------------------------------------
# Mixed Gaussian second-moment interval plots
# ---------------------------------------------------------------------


def plot_mixed_moment_widths(summary: pd.DataFrame, output_dir: Path) -> None:
    """Gaussian moment widths and residual second moments in the full-set setting."""

    required = {"width_gaussian_fixed_target", "M_fixed_target"}
    if not required.issubset(set(summary.columns)):
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    gamma_width_target = _best_gamma(summary, "width_gaussian_fixed_target")
    gamma_moment_target = _best_gamma(summary, "M_fixed_target")

    # Widths
    ax = axes[0]
    ax.plot(
        summary["gamma"],
        summary["width_gaussian_fixed_target"],
        color=COLORS["teal"],
        linewidth=2.2,
        linestyle="-",
        label="Gaussian moment, fixed target",
    )

    if "width_gaussian_own_set" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["width_gaussian_own_set"],
            color=STYLES["matched_moment"]["color"],
            linewidth=STYLES["matched_moment"]["linewidth"],
            linestyle=STYLES["matched_moment"]["linestyle"],
            label=r"Gaussian moment, matched-$\gamma$ set",
        )

    if "width_std_calibration" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["width_std_calibration"],
            **STYLES["calibration_quantile"],
        )

    if gamma_width_target is not None:
        _vline(ax, gamma_width_target, r"$\gamma^\star_{\mathrm{width},\eta}$", y_pos=0.93)

    _apply_gamma_focus(ax, summary, xmax=MIXED_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["width_gaussian_fixed_target", "width_gaussian_own_set", "width_std_calibration"],
        xmax=MIXED_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Interval width")
    ax.grid(True, alpha=0.22)

    # Moments / risks
    ax = axes[1]
    ax.plot(
        summary["gamma"],
        summary["M_fixed_target"],
        color=COLORS["teal"],
        linewidth=2.2,
        linestyle="-",
        label="fixed target moment",
    )

    if "M_own_set" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["M_own_set"],
            color=STYLES["matched_moment"]["color"],
            linewidth=STYLES["matched_moment"]["linewidth"],
            linestyle=STYLES["matched_moment"]["linestyle"],
            label=r"matched-$\gamma$ moment",
        )

    if "risk_obs_max" in summary.columns:
        ax.plot(
            summary["gamma"],
            summary["risk_obs_max"],
            **STYLES["max_observed_risk"],
        )

    if gamma_moment_target is not None:
        _vline(ax, gamma_moment_target, r"$\gamma^\star_{\mathrm{risk},\eta}$", y_pos=0.93)

    _apply_gamma_focus(ax, summary, xmax=MIXED_GAMMA_MAX)

    vals = _values_in_focus(
        summary,
        ["M_fixed_target", "M_own_set", "risk_obs_max"],
        xmax=MIXED_GAMMA_MAX,
    )
    _set_zoomed_ylim(ax, vals, lower_floor=0.0, pad_frac=0.10)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Residual second moment / risk")
    ax.grid(True, alpha=0.22)

    _axis_legend_above(axes[0], ncol=1, y=1.01)
    _axis_legend_above(axes[1], ncol=1, y=1.01)
    _save(fig, output_dir, "mixed_moment_widths", rect=(0.0, 0.0, 1.0, 0.80))


def plot_mixed_tv_vacuity(summary: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    """TV radius and clipped RV quantile level for the full second-moment set."""

    rho_col = "rho_full_set_TV_worst"
    if rho_col not in summary.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # Full-set TV radius
    ax = axes[0]
    ax.plot(
        summary["gamma"],
        summary[rho_col],
        color=COLORS["black"],
        linewidth=2.1,
        linestyle="-",
    )
    _hline(ax, 2.0 * alpha, r"$2\alpha$")
    ax.set_ylim(0.0, 2.05)
    _apply_gamma_focus(ax, summary, xmax=MIXED_GAMMA_MAX)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"TV radius $\rho$")
    ax.grid(True, alpha=0.22)

    # Robust quantile level
    ax = axes[1]
    u = np.minimum(1.0 - alpha + summary[rho_col].to_numpy(dtype=float) / 2.0, 1.0)
    ax.plot(
        summary["gamma"],
        u,
        color=COLORS["rust"],
        linewidth=2.1,
        linestyle="-",
    )
    _hline(ax, 1.0, "1")
    ax.set_ylim(max(0.0, 1.0 - alpha - 0.05), 1.01)
    _apply_gamma_focus(ax, summary, xmax=MIXED_GAMMA_MAX)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("RV quantile level")
    ax.grid(True, alpha=0.22)

    _save(fig, output_dir, "mixed_tv_vacuity")


def plot_mixed_residuals(details: pd.DataFrame, output_dir: Path) -> None:
    """Observed residual means and standard deviations in the full-set setting."""

    observed = details[details["kind"] == "observed"].copy()
    env_colors = [COLORS["blue"], COLORS["rust"], COLORS["teal"], COLORS["purple"], COLORS["gold"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    for i, (name, group) in enumerate(observed.groupby("name", sort=False)):
        group = group.sort_values("gamma")

        axes[0].plot(
            group["gamma"],
            group["mu"],
            color=env_colors[i % len(env_colors)],
            linewidth=1.8,
            label=_pretty_label(name),
        )

        axes[1].plot(
            group["gamma"],
            group["sigma"],
            color=env_colors[i % len(env_colors)],
            linewidth=1.8,
            label=_pretty_label(name),
        )

    axes[0].set_ylabel("Observed residual mean")
    axes[1].set_ylabel("Observed residual standard deviation")

    for ax in axes:
        ax.set_xlabel(r"$\gamma$")
        ax.grid(True, alpha=0.22)

    dummy_summary = pd.DataFrame({"gamma": observed["gamma"].unique()})
    for ax in axes:
        _apply_gamma_focus(ax, dummy_summary, xmax=MIXED_GAMMA_MAX)

    _shared_legend(fig, axes, ncol=3, y=1.00)
    _save(fig, output_dir, "mixed_residual_diagnostics", rect=(0.0, 0.0, 1.0, 0.91))


def make_mixed_plots(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    alpha: float,
) -> None:
    plot_mixed_moment_widths(summary, output_dir)
    plot_mixed_tv_vacuity(summary, output_dir, alpha)
    plot_mixed_residuals(details, output_dir)