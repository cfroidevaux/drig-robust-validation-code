"""Reproduce the numerical population study from Chapter 4.

Running this script creates the CSV files and figures used in the thesis.
The default settings write deterministic outputs to outputs/<setting>/latest/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import (
    GammaGrid,
    analyze_cov_only,
    analyze_mean_only,
    analyze_mixed_second_moment,
    finite_minimizer,
)
from src.plotting import make_mixed_plots, make_structured_plots
from src.settings import CovOnlySetting, MeanOnlySetting, MixedSetting, SETTINGS


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "inf"
    return f"{x:.6g}"


def _valid(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[summary_df["valid"] == True].copy()


def _structured_metrics(setting, summary_df: pd.DataFrame, kind: str) -> dict[str, Any]:
    valid = _valid(summary_df)
    if valid.empty:
        return {"setting": setting.name, "kind": kind, "valid_gamma_count": 0}
    best_width = finite_minimizer(summary_df, "width_rv")
    best_rho = finite_minimizer(summary_df, "rho_wc")
    best_target = finite_minimizer(summary_df, "risk_fixed_target")
    best_fixed_target_width = finite_minimizer(summary_df, "width_rv_fixed_target")
    row0 = valid.iloc[0]
    finite_width = valid[np.isfinite(valid["width_rv"].astype(float))]
    first_finite_gamma = float(finite_width.iloc[0]["gamma"]) if not finite_width.empty else np.nan
    row_width = valid.loc[valid["gamma"].sub(best_width["gamma"]).abs().idxmin()] if np.isfinite(best_width["gamma"]) else None
    return {
        "setting": setting.name,
        "kind": kind,
        "p": setting.p,
        "alpha": setting.alpha,
        "target_gamma": float(getattr(setting, "target_gamma", np.nan)),
        "valid_gamma_count": len(valid),
        "rho_at_gamma0": float(row0["rho_wc"]),
        "first_finite_rv_gamma": first_finite_gamma,
        "gamma_width_star": best_width["gamma"],
        "min_width_rv": best_width["width_rv"],
        "gamma_rho_star": best_rho["gamma"],
        "min_rho_wc": best_rho["rho_wc"],
        "gamma_fixed_target_star": best_target["gamma"],
        "min_risk_fixed_target": best_target["risk_fixed_target"],
        "coverage_std_wc_at_width_star": float(row_width["coverage_std_wc"]) if row_width is not None else np.nan,
        "coverage_rv_wc_at_width_star": float(row_width["coverage_rv_wc"]) if row_width is not None else np.nan,
        "risk_fixed_target_at_width_star": float(row_width["risk_fixed_target"]) if row_width is not None else np.nan,
        "gamma_width_fixed_target_star": best_fixed_target_width["gamma"],
        "min_width_rv_fixed_target": best_fixed_target_width["width_rv_fixed_target"],
    }


def _mixed_metrics(setting, summary_df: pd.DataFrame, kind: str) -> dict[str, Any]:
    valid = _valid(summary_df)
    if valid.empty:
        return {"setting": setting.name, "kind": kind, "valid_gamma_count": 0}
    best_target_width = finite_minimizer(summary_df, "width_gaussian_fixed_target")
    best_target_moment = finite_minimizer(summary_df, "M_fixed_target")
    best_own_width = finite_minimizer(summary_df, "width_gaussian_own_set")
    row0 = valid.iloc[0]
    return {
        "setting": setting.name,
        "kind": kind,
        "p": setting.p,
        "alpha": setting.alpha,
        "target_gamma": float(getattr(setting, "target_gamma", np.nan)),
        "valid_gamma_count": len(valid),
        "gamma_target_width_star": best_target_width["gamma"],
        "min_width_gaussian_fixed_target": best_target_width["width_gaussian_fixed_target"],
        "gamma_target_moment_star": best_target_moment["gamma"],
        "min_M_fixed_target": best_target_moment["M_fixed_target"],
        "gamma_own_width_star": best_own_width["gamma"],
        "min_width_gaussian_own_set": best_own_width["width_gaussian_own_set"],
        "width_gaussian_fixed_target_at_gamma0": float(row0["width_gaussian_fixed_target"]),
        "width_gaussian_own_set_at_gamma0": float(row0["width_gaussian_own_set"]),
        "rho_full_set_TV_worst": 2.0,
    }


def _write_summary(setting, summary_df: pd.DataFrame, out_dir: Path, kind: str) -> None:
    lines: list[str] = []
    lines.append(f"Setting: {setting.name}")
    lines.append(f"Kind: {kind}")
    lines.append(f"Description: {setting.description}")
    lines.append(f"p = {setting.p}")
    lines.append(f"alpha = {setting.alpha}")
    if hasattr(setting, "target_gamma"):
        lines.append(f"fixed target severity eta = {getattr(setting, 'target_gamma')}")
    valid = _valid(summary_df)
    lines.append(f"Valid gamma values: {len(valid)} / {len(summary_df)}")

    if kind in {"mean_only", "cov_only"} and not valid.empty:
        metrics = _structured_metrics(setting, summary_df, kind)
        lines.append("")
        lines.append("TV-RV worst-case over structured Gaussian subclass:")
        for key in [
            "rho_at_gamma0",
            "first_finite_rv_gamma",
            "gamma_width_star",
            "min_width_rv",
            "gamma_rho_star",
            "min_rho_wc",
            "gamma_fixed_target_star",
            "min_risk_fixed_target",
            "coverage_std_wc_at_width_star",
            "coverage_rv_wc_at_width_star",
            "risk_fixed_target_at_width_star",
        ]:
            lines.append(f"  {key} = {_fmt(float(metrics[key]))}")
    elif kind == "mixed" and not valid.empty:
        metrics = _mixed_metrics(setting, summary_df, kind)
        lines.append("")
        lines.append("Gaussian second-moment intervals:")
        for key in [
            "gamma_target_width_star",
            "min_width_gaussian_fixed_target",
            "gamma_target_moment_star",
            "min_M_fixed_target",
            "gamma_own_width_star",
            "min_width_gaussian_own_set",
            "width_gaussian_fixed_target_at_gamma0",
            "width_gaussian_own_set_at_gamma0",
            "rho_full_set_TV_worst",
        ]:
            lines.append(f"  {key} = {_fmt(float(metrics[key]))}")
        lines.append("  Full-set TV-RV remains vacuous by Proposition 3.6.")

    (out_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(setting_name: str, args) -> tuple[Path, dict[str, Any]]:
    setting = SETTINGS[setting_name]
    gamma_values = GammaGrid(args.gamma_start, args.gamma_stop, args.gamma_step).values()

    timestamp = "latest" if args.no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / setting.name / timestamp
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(setting, MeanOnlySetting):
        summary, details = analyze_mean_only(setting, gamma_values, tv_grid_n=args.tv_grid_n, opt_grid_n=args.opt_grid_n)
        kind = "mean_only"
        make_structured_plots(summary, details, plot_dir, setting.alpha, setting.weights, parameter="mean")
        metrics = _structured_metrics(setting, summary, kind)
    elif isinstance(setting, CovOnlySetting):
        summary, details = analyze_cov_only(setting, gamma_values, tv_grid_n=args.tv_grid_n, opt_grid_n=args.opt_grid_n)
        kind = "cov_only"
        make_structured_plots(summary, details, plot_dir, setting.alpha, setting.weights, parameter="cov")
        metrics = _structured_metrics(setting, summary, kind)
    elif isinstance(setting, MixedSetting):
        summary, details = analyze_mixed_second_moment(setting, gamma_values)
        kind = "mixed"
        make_mixed_plots(summary, details, plot_dir, setting.alpha)
        metrics = _mixed_metrics(setting, summary, kind)
    else:
        raise TypeError(f"Unsupported setting type: {type(setting)}")

    summary.to_csv(table_dir / "summary_by_gamma.csv", index=False)
    details.to_csv(table_dir / "residual_details_by_gamma.csv", index=False)

    meta = {
        "setting": setting.name,
        "kind": kind,
        "description": setting.description,
        "p": setting.p,
        "alpha": setting.alpha,
        "target_gamma": float(getattr(setting, "target_gamma", np.nan)),
        "gamma_start": args.gamma_start,
        "gamma_stop": args.gamma_stop,
        "gamma_step": args.gamma_step,
        "tv_grid_n": args.tv_grid_n,
        "opt_grid_n": args.opt_grid_n,
        "created_at": timestamp,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_summary(setting, summary, out_dir, kind)
    return out_dir, metrics


def write_global_summary(metrics: list[dict[str, Any]], output_dir: str, timestamp: str) -> None:
    out = Path(output_dir) / "_summary" / timestamp
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(out / "chapter4_setting_summary.csv", index=False)
    # A compact LaTeX table for quick copy/paste; users can edit captions/formatting later.
    (out / "chapter4_setting_summary.tex").write_text(df.to_latex(index=False, float_format="%.4g"), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Chapter 4 uncertainty-set analyses")
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS.keys()), choices=list(SETTINGS.keys()))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-stop", type=float, default=15.0)
    parser.add_argument("--gamma-step", type=float, default=0.01)
    parser.add_argument("--tv-grid-n", type=int, default=1400)
    parser.add_argument("--opt-grid-n", type=int, default=151)
    parser.add_argument("--no-timestamp", action="store_true", default=True, help="Write into outputs/<setting>/latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = "latest" if args.no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics: list[dict[str, Any]] = []
    for setting_name in args.settings:
        out_dir, metrics = run_one(setting_name, 
                                   args)
        all_metrics.append(metrics)
        print(f"Finished {setting_name}. Results written to {out_dir}")
    write_global_summary(all_metrics, args.output_dir, timestamp)
    print(f"Combined summary written to {Path(args.output_dir) / '_summary' / timestamp}")


if __name__ == "__main__":
    main()
