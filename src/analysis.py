"""Analysis routines for the Chapter 4 uncertainty-set population study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .scm_core import observed_residual_laws, solve_b_gamma
from .score_tools import (
    ResidualLaw,
    folded_normal_cdf,
    integration_upper_bound,
    mixture_pdf,
    residual_law_from_row,
    robust_validation_quantities,
    score_coverage,
    tv_radius_grid,
)
from .settings import CovOnlySetting, MeanOnlySetting, MixedSetting
from .uncertainty_sets import (
    Delta_obs_cov,
    M_obs_mean,
    S_gamma,
    cov_only_h_max,
    gaussian_moment_halfwidth,
    mean_only_m_max,
)


@dataclass(frozen=True)
class GammaGrid:
    """Gamma grid configuration."""

    start: float = 0.0
    stop: float = 15.0
    step: float = 0.25

    def values(self) -> np.ndarray:
        n = int(round((self.stop - self.start) / self.step))
        return np.round(np.linspace(self.start, self.stop, n + 1), 10)


def _calibration_laws(obs_rows: List[Dict[str, float]]) -> List[ResidualLaw]:
    return [residual_law_from_row(row) for row in obs_rows]


def _tv_upper(cal_laws: List[ResidualLaw], extra_laws: List[ResidualLaw]) -> float:
    return integration_upper_bound([*cal_laws, *extra_laws], multiplier=12.0)


def _vector_columns(prefix: str, x: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{j+1}": float(x[j]) for j in range(len(x))}


def _worst_mean_tv_grid(
    cal_laws: List[ResidualLaw],
    weights: np.ndarray,
    sigma: float,
    m_max: float,
    tv_grid_n: int,
    opt_grid_n: int,
) -> Tuple[float, float]:
    """Grid maximization of TV over m in [0, m_max]."""

    if m_max <= 1e-14:
        law = ResidualLaw("tv_worst_mean", 0.0, sigma)
        return 0.0, tv_radius_grid(cal_laws, weights, law, grid_n=tv_grid_n)

    m_vals = np.linspace(0.0, m_max, opt_grid_n)
    upper = _tv_upper(cal_laws, [ResidualLaw("upper", m_max, sigma)])
    x = np.linspace(0.0, upper, tv_grid_n)
    cal_pdf = mixture_pdf(x, cal_laws, weights)

    z1 = (x[None, :] - m_vals[:, None]) / sigma
    z2 = (x[None, :] + m_vals[:, None]) / sigma
    test_pdf = (norm.pdf(z1) + norm.pdf(z2)) / sigma
    rhos = np.trapz(np.abs(test_pdf - cal_pdf[None, :]), x, axis=1)
    idx = int(np.argmax(rhos))
    return float(m_vals[idx]), float(min(max(rhos[idx], 0.0), 2.0))


def _worst_cov_tv_grid(
    cal_laws: List[ResidualLaw],
    weights: np.ndarray,
    mean: float,
    sigma0_sq: float,
    h_max: float,
    tv_grid_n: int,
    opt_grid_n: int,
) -> Tuple[float, float]:
    """Grid maximization of TV over h in [0, h_max]."""

    if h_max <= 1e-14:
        law = ResidualLaw("tv_worst_cov", mean, float(np.sqrt(sigma0_sq)))
        return 0.0, tv_radius_grid(cal_laws, weights, law, grid_n=tv_grid_n)

    h_vals = np.linspace(0.0, h_max, opt_grid_n)
    sigmas = np.sqrt(sigma0_sq + h_vals)
    upper = _tv_upper(cal_laws, [ResidualLaw("upper", mean, float(np.max(sigmas)))])
    x = np.linspace(0.0, upper, tv_grid_n)
    cal_pdf = mixture_pdf(x, cal_laws, weights)

    z1 = (x[None, :] - mean) / sigmas[:, None]
    z2 = (x[None, :] + mean) / sigmas[:, None]
    test_pdf = (norm.pdf(z1) + norm.pdf(z2)) / sigmas[:, None]
    rhos = np.trapz(np.abs(test_pdf - cal_pdf[None, :]), x, axis=1)
    idx = int(np.argmax(rhos))
    return float(h_vals[idx]), float(min(max(rhos[idx], 0.0), 2.0))


def _worst_mean_coverage_grid(q: float, sigma: float, m_max: float, opt_grid_n: int) -> tuple[float, float]:
    """Minimize coverage over m in [0,m_max] for a fixed interval half-width q."""

    if np.isinf(q):
        return 0.0, 1.0
    m_vals = np.linspace(0.0, max(m_max, 0.0), max(opt_grid_n, 2))
    cov = norm.cdf((q - m_vals) / sigma) - norm.cdf((-q - m_vals) / sigma)
    idx = int(np.argmin(cov))
    return float(m_vals[idx]), float(cov[idx])


def _worst_cov_coverage_grid(q: float, mean: float, sigma0_sq: float, h_max: float, opt_grid_n: int) -> tuple[float, float]:
    """Minimize coverage over h in [0,h_max] for a fixed interval half-width q."""

    if np.isinf(q):
        return 0.0, 1.0
    h_vals = np.linspace(0.0, max(h_max, 0.0), max(opt_grid_n, 2))
    sigmas = np.sqrt(sigma0_sq + h_vals)
    cov = norm.cdf((q - mean) / sigmas) - norm.cdf((-q - mean) / sigmas)
    idx = int(np.argmin(cov))
    return float(h_vals[idx]), float(cov[idx])


def analyze_mean_only(
    setting: MeanOnlySetting,
    gamma_values: np.ndarray,
    alpha: float | None = None,
    tv_grid_n: int = 2500,
    opt_grid_n: int = 251,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze TV-RV worst case over the mean-only Gaussian subclass."""

    alpha = setting.alpha if alpha is None else alpha
    Mobs = M_obs_mean(setting)
    target_gamma = float(setting.target_gamma)
    summary_rows: List[Dict[str, float | str | bool]] = []
    detail_rows: List[Dict[str, float | str]] = []

    for gamma in gamma_values:
        core = solve_b_gamma(setting, float(gamma))
        row_base = {"gamma": float(gamma), "A_min_eig": core["A_min_eig"]}
        if not core["A_positive_definite"]:
            summary_rows.append({**row_base, "valid": False})
            continue

        b = core["b"]
        w = core["w"]
        K = core["K"]
        obs_rows = observed_residual_laws(setting, K, b)
        cal_laws = _calibration_laws(obs_rows)

        sigma2 = float(w.T @ setting.Omega @ w)
        sigma = float(np.sqrt(sigma2))
        projected_penalty = float(w.T @ Mobs @ w)
        m_max = mean_only_m_max(w, Mobs, float(gamma))
        m_target_max = float(np.sqrt(max(0.0, target_gamma * projected_penalty)))
        m_star_tv, rho_wc = _worst_mean_tv_grid(cal_laws, setting.weights, sigma, m_max, tv_grid_n, opt_grid_n)
        m_star_target_tv, rho_fixed_target = _worst_mean_tv_grid(
            cal_laws,
            setting.weights,
            sigma,
            m_target_max,
            tv_grid_n,
            opt_grid_n,
        )

        tv_law = ResidualLaw("tv_worst_mean", m_star_tv, sigma)
        rv = robust_validation_quantities(cal_laws, setting.weights, alpha, rho_wc)
        rv_fixed_target = robust_validation_quantities(
            cal_laws,
            setting.weights,
            alpha,
            rho_fixed_target,
        )
        coverage_std_tvmax = score_coverage(rv["q_std"], tv_law)
        coverage_rv_tvmax = score_coverage(rv["q_rv"], tv_law)
        m_star_cov_std, coverage_std_wc = _worst_mean_coverage_grid(rv["q_std"], sigma, m_max, opt_grid_n)
        m_star_cov_rv, coverage_rv_wc = _worst_mean_coverage_grid(rv["q_rv"], sigma, m_max, opt_grid_n)
        m_star_cov_fixed_std, coverage_std_fixed_target = _worst_mean_coverage_grid(
            rv["q_std"],
            sigma,
            m_target_max,
            opt_grid_n,
        )
        m_star_cov_fixed_current_rv, coverage_current_rv_on_fixed_target = _worst_mean_coverage_grid(
            rv["q_rv"],
            sigma,
            m_target_max,
            opt_grid_n,
        )
        m_star_cov_fixed_target_rv, coverage_target_rv_fixed_target = _worst_mean_coverage_grid(
            rv_fixed_target["q_rv"],
            sigma,
            m_target_max,
            opt_grid_n,
        )

        risk_current_set = sigma2 + m_max**2
        risk_fixed_target = sigma2 + m_target_max**2
        risk_tv_law = sigma2 + m_star_tv**2

        summary_rows.append(
            {
                **row_base,
                "valid": True,
                **_vector_columns("b", b),
                **_vector_columns("w", w),
                "mean_penalty": projected_penalty,
                "gamma_mean_penalty": float(gamma) * projected_penalty,
                "target_gamma": target_gamma,
                "m_max": m_max,
                "m_target_max": m_target_max,
                "worst_m_tv": m_star_tv,
                "coverage_worst_m_std": m_star_cov_std,
                "coverage_worst_m_rv": m_star_cov_rv,
                "sigma": sigma,
                **rv,
                "coverage_std_tvmax": coverage_std_tvmax,
                "coverage_rv_tvmax": coverage_rv_tvmax,
                "coverage_std_wc": coverage_std_wc,
                "coverage_rv_wc": coverage_rv_wc,
                "risk_obs_max": max(float(r["risk"]) for r in obs_rows),
                "risk_current_set": risk_current_set,
                "risk_fixed_target": risk_fixed_target,
                "risk_tv_law": risk_tv_law,
                "rho_fixed_target": rho_fixed_target,
                "worst_m_fixed_target_tv": m_star_target_tv,
                "u_rv_fixed_target": rv_fixed_target["u_rv"],
                "q_rv_fixed_target": rv_fixed_target["q_rv"],
                "width_rv_fixed_target": rv_fixed_target["width_rv"],
                "coverage_std_fixed_target": coverage_std_fixed_target,
                "coverage_current_rv_on_fixed_target": coverage_current_rv_on_fixed_target,
                "coverage_target_rv_fixed_target": coverage_target_rv_fixed_target,
                "coverage_worst_m_fixed_std": m_star_cov_fixed_std,
                "coverage_worst_m_fixed_current_rv": m_star_cov_fixed_current_rv,
                "coverage_worst_m_fixed_target_rv": m_star_cov_fixed_target_rv,
            }
        )
        for obs in obs_rows:
            detail_rows.append({"gamma": float(gamma), "kind": "observed", **obs})
        detail_rows.extend(
            [
                {
                    "gamma": float(gamma),
                    "kind": "tv_worst_test",
                    "name": "tv_worst_mean",
                    "mu": m_star_tv,
                    "var": sigma2,
                    "sigma": sigma,
                    "risk": risk_tv_law,
                },
                {
                    "gamma": float(gamma),
                    "kind": "boundary_current_set",
                    "name": "boundary_mean_current",
                    "mu": m_max,
                    "var": sigma2,
                    "sigma": sigma,
                    "risk": risk_current_set,
                },
                {
                    "gamma": float(gamma),
                    "kind": "boundary_fixed_target",
                    "name": "boundary_mean_target",
                    "mu": m_target_max,
                    "var": sigma2,
                    "sigma": sigma,
                    "risk": risk_fixed_target,
                },
                {
                    "gamma": float(gamma),
                    "kind": "tv_worst_fixed_target",
                    "name": "tv_worst_mean_fixed_target",
                    "mu": m_star_target_tv,
                    "var": sigma2,
                    "sigma": sigma,
                    "risk": sigma2 + m_star_target_tv**2,
                },
            ]
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def analyze_cov_only(
    setting: CovOnlySetting,
    gamma_values: np.ndarray,
    alpha: float | None = None,
    tv_grid_n: int = 2500,
    opt_grid_n: int = 251,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze TV-RV worst case over the covariance-only Gaussian subclass."""

    alpha = setting.alpha if alpha is None else alpha
    Delta = Delta_obs_cov(setting)
    target_gamma = float(setting.target_gamma)
    summary_rows: List[Dict[str, float | str | bool]] = []
    detail_rows: List[Dict[str, float | str]] = []

    for gamma in gamma_values:
        core = solve_b_gamma(setting, float(gamma))
        row_base = {"gamma": float(gamma), "A_min_eig": core["A_min_eig"]}
        if not core["A_positive_definite"]:
            summary_rows.append({**row_base, "valid": False})
            continue

        b = core["b"]
        w = core["w"]
        K = core["K"]
        obs_rows = observed_residual_laws(setting, K, b)
        cal_laws = _calibration_laws(obs_rows)

        m = float(w.T @ setting.delta_common)
        sigma0_sq = float(w.T @ setting.Omega0 @ w)
        projected_penalty = float(w.T @ Delta @ w)
        h_max = cov_only_h_max(w, Delta, float(gamma))
        h_target_max = max(0.0, target_gamma * projected_penalty)
        h_star_tv, rho_wc = _worst_cov_tv_grid(cal_laws, setting.weights, m, sigma0_sq, h_max, tv_grid_n, opt_grid_n)
        h_star_target_tv, rho_fixed_target = _worst_cov_tv_grid(
            cal_laws,
            setting.weights,
            m,
            sigma0_sq,
            h_target_max,
            tv_grid_n,
            opt_grid_n,
        )

        tv_sigma = float(np.sqrt(sigma0_sq + h_star_tv))
        tv_law = ResidualLaw("tv_worst_cov", m, tv_sigma)
        rv = robust_validation_quantities(cal_laws, setting.weights, alpha, rho_wc)
        rv_fixed_target = robust_validation_quantities(
            cal_laws,
            setting.weights,
            alpha,
            rho_fixed_target,
        )
        coverage_std_tvmax = score_coverage(rv["q_std"], tv_law)
        coverage_rv_tvmax = score_coverage(rv["q_rv"], tv_law)
        h_star_cov_std, coverage_std_wc = _worst_cov_coverage_grid(rv["q_std"], m, sigma0_sq, h_max, opt_grid_n)
        h_star_cov_rv, coverage_rv_wc = _worst_cov_coverage_grid(rv["q_rv"], m, sigma0_sq, h_max, opt_grid_n)
        h_star_cov_fixed_std, coverage_std_fixed_target = _worst_cov_coverage_grid(
            rv["q_std"],
            m,
            sigma0_sq,
            h_target_max,
            opt_grid_n,
        )
        h_star_cov_fixed_current_rv, coverage_current_rv_on_fixed_target = _worst_cov_coverage_grid(
            rv["q_rv"],
            m,
            sigma0_sq,
            h_target_max,
            opt_grid_n,
        )
        h_star_cov_fixed_target_rv, coverage_target_rv_fixed_target = _worst_cov_coverage_grid(
            rv_fixed_target["q_rv"],
            m,
            sigma0_sq,
            h_target_max,
            opt_grid_n,
        )

        risk_current_set = m**2 + sigma0_sq + h_max
        risk_fixed_target = m**2 + sigma0_sq + h_target_max
        risk_tv_law = m**2 + sigma0_sq + h_star_tv

        summary_rows.append(
            {
                **row_base,
                "valid": True,
                **_vector_columns("b", b),
                **_vector_columns("w", w),
                "cov_penalty": projected_penalty,
                "gamma_cov_penalty": float(gamma) * projected_penalty,
                "target_gamma": target_gamma,
                "h_max": h_max,
                "h_target_max": h_target_max,
                "worst_h_tv": h_star_tv,
                "coverage_worst_h_std": h_star_cov_std,
                "coverage_worst_h_rv": h_star_cov_rv,
                "mean_common_projected": m,
                "sigma0": float(np.sqrt(sigma0_sq)),
                "sigma_tv_worst": tv_sigma,
                **rv,
                "coverage_std_tvmax": coverage_std_tvmax,
                "coverage_rv_tvmax": coverage_rv_tvmax,
                "coverage_std_wc": coverage_std_wc,
                "coverage_rv_wc": coverage_rv_wc,
                "risk_obs_max": max(float(r["risk"]) for r in obs_rows),
                "risk_current_set": risk_current_set,
                "risk_fixed_target": risk_fixed_target,
                "risk_tv_law": risk_tv_law,
                "rho_fixed_target": rho_fixed_target,
                "worst_h_fixed_target_tv": h_star_target_tv,
                "u_rv_fixed_target": rv_fixed_target["u_rv"],
                "q_rv_fixed_target": rv_fixed_target["q_rv"],
                "width_rv_fixed_target": rv_fixed_target["width_rv"],
                "coverage_std_fixed_target": coverage_std_fixed_target,
                "coverage_current_rv_on_fixed_target": coverage_current_rv_on_fixed_target,
                "coverage_target_rv_fixed_target": coverage_target_rv_fixed_target,
                "coverage_worst_h_fixed_std": h_star_cov_fixed_std,
                "coverage_worst_h_fixed_current_rv": h_star_cov_fixed_current_rv,
                "coverage_worst_h_fixed_target_rv": h_star_cov_fixed_target_rv,
            }
        )
        for obs in obs_rows:
            detail_rows.append({"gamma": float(gamma), "kind": "observed", **obs})
        detail_rows.extend(
            [
                {
                    "gamma": float(gamma),
                    "kind": "tv_worst_test",
                    "name": "tv_worst_cov",
                    "mu": m,
                    "var": sigma0_sq + h_star_tv,
                    "sigma": tv_sigma,
                    "risk": risk_tv_law,
                },
                {
                    "gamma": float(gamma),
                    "kind": "boundary_current_set",
                    "name": "boundary_cov_current",
                    "mu": m,
                    "var": sigma0_sq + h_max,
                    "sigma": float(np.sqrt(sigma0_sq + h_max)),
                    "risk": risk_current_set,
                },
                {
                    "gamma": float(gamma),
                    "kind": "boundary_fixed_target",
                    "name": "boundary_cov_target",
                    "mu": m,
                    "var": sigma0_sq + h_target_max,
                    "sigma": float(np.sqrt(sigma0_sq + h_target_max)),
                    "risk": risk_fixed_target,
                },
                {
                    "gamma": float(gamma),
                    "kind": "tv_worst_fixed_target",
                    "name": "tv_worst_cov_fixed_target",
                    "mu": m,
                    "var": sigma0_sq + h_star_target_tv,
                    "sigma": float(np.sqrt(sigma0_sq + h_star_target_tv)),
                    "risk": m**2 + sigma0_sq + h_star_target_tv,
                },
            ]
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def analyze_mixed_second_moment(
    setting: MixedSetting,
    gamma_values: np.ndarray,
    alpha: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze Gaussian second-moment intervals for the mixed setting."""

    alpha = setting.alpha if alpha is None else alpha
    target_gamma = float(setting.target_gamma)
    S_target = S_gamma(setting, target_gamma)
    summary_rows: List[Dict[str, float | str | bool]] = []
    detail_rows: List[Dict[str, float | str]] = []

    for gamma in gamma_values:
        core = solve_b_gamma(setting, float(gamma))
        row_base = {"gamma": float(gamma), "A_min_eig": core["A_min_eig"]}
        if not core["A_positive_definite"]:
            summary_rows.append({**row_base, "valid": False})
            continue

        b = core["b"]
        w = core["w"]
        K = core["K"]
        obs_rows = observed_residual_laws(setting, K, b)
        cal_laws = _calibration_laws(obs_rows)
        rv_cal_only = robust_validation_quantities(cal_laws, setting.weights, alpha, rho_wc=0.0)

        S_own = S_gamma(setting, float(gamma))
        M_own = float(w.T @ S_own @ w)
        M_target = float(w.T @ S_target @ w)
        q_own = gaussian_moment_halfwidth(w, S_own, alpha)
        q_target = gaussian_moment_halfwidth(w, S_target, alpha)

        summary_rows.append(
            {
                **row_base,
                "valid": True,
                **_vector_columns("b", b),
                **_vector_columns("w", w),
                "target_gamma": target_gamma,
                "M_own_set": M_own,
                "M_fixed_target": M_target,
                "risk_current_set": M_own,
                "risk_fixed_target": M_target,
                "q_gaussian_own_set": q_own,
                "width_gaussian_own_set": 2.0 * q_own,
                "q_gaussian_fixed_target": q_target,
                "width_gaussian_fixed_target": 2.0 * q_target,
                "q_std_calibration": rv_cal_only["q_std"],
                "width_std_calibration": rv_cal_only["width_std"],
                "risk_obs_max": max(float(r["risk"]) for r in obs_rows),
                "rho_full_set_TV_worst": 2.0,
            }
        )
        for obs in obs_rows:
            detail_rows.append({"gamma": float(gamma), "kind": "observed", **obs})

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def finite_minimizer(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Return grid minimizer of a finite numeric column."""

    valid = df[(df["valid"] == True) & np.isfinite(df[column].astype(float))].copy()
    if valid.empty:
        return {"gamma": np.nan, column: np.nan}
    idx = valid[column].astype(float).idxmin()
    return {"gamma": float(valid.loc[idx, "gamma"]), column: float(valid.loc[idx, column])}
