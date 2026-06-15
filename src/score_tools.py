"""Folded-normal score laws and population robust-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class ResidualLaw:
    """Gaussian residual law U ~ N(mu, sigma^2)."""

    name: str
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("ResidualLaw sigma must be positive")


def folded_normal_cdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    """CDF of |N(mu, sigma^2)| on [0, infinity)."""

    x_arr = np.asarray(x)
    val = norm.cdf((x_arr - mu) / sigma) - norm.cdf((-x_arr - mu) / sigma)
    val = np.where(x_arr < 0, 0.0, val)
    if np.isscalar(x):
        return float(val)
    return val


def folded_normal_pdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    """Density of |N(mu, sigma^2)| on [0, infinity)."""

    x_arr = np.asarray(x)
    val = (norm.pdf((x_arr - mu) / sigma) + norm.pdf((x_arr + mu) / sigma)) / sigma
    val = np.where(x_arr < 0, 0.0, val)
    if np.isscalar(x):
        return float(val)
    return val


def mixture_cdf(x: np.ndarray | float, laws: List[ResidualLaw], weights: np.ndarray) -> np.ndarray | float:
    """CDF of weighted mixture of folded-normal score laws."""

    weights = np.asarray(weights, dtype=float)
    total = 0.0
    for law, weight in zip(laws, weights):
        total = total + weight * folded_normal_cdf(x, law.mu, law.sigma)
    return total


def mixture_pdf(x: np.ndarray | float, laws: List[ResidualLaw], weights: np.ndarray) -> np.ndarray | float:
    """PDF of weighted mixture of folded-normal score laws."""

    weights = np.asarray(weights, dtype=float)
    total = 0.0
    for law, weight in zip(laws, weights):
        total = total + weight * folded_normal_pdf(x, law.mu, law.sigma)
    return total


def integration_upper_bound(laws: Iterable[ResidualLaw], multiplier: float = 12.0) -> float:
    """Heuristic upper bound covering folded-normal tails."""

    vals = [abs(law.mu) + multiplier * law.sigma for law in laws]
    return float(max(max(vals), 1.0))


def adaptive_quantile(cdf: Callable[[float], float], level: float, initial_upper: float = 10.0) -> float:
    """Return inf{x>=0: cdf(x)>=level}. Returns inf for level>=1."""

    if level >= 1.0:
        return float("inf")
    if level <= 0.0:
        return 0.0
    upper = max(float(initial_upper), 1.0)
    while cdf(upper) < level:
        upper *= 2.0
        if upper > 1e10:
            raise RuntimeError(f"Could not bracket quantile at level={level}")
    return float(brentq(lambda z: cdf(z) - level, 0.0, upper, xtol=1e-10, rtol=1e-10))


def tv_radius_grid(
    cal_laws: List[ResidualLaw],
    weights: np.ndarray,
    test_law: ResidualLaw,
    grid_n: int = 6000,
    upper: float | None = None,
) -> float:
    """Compute rho = int |f_test - f_cal| on [0, infinity) by trapezoidal grid."""

    if upper is None:
        upper = integration_upper_bound([*cal_laws, test_law], multiplier=12.0)
    x = np.linspace(0.0, upper, grid_n)
    diff = np.abs(folded_normal_pdf(x, test_law.mu, test_law.sigma) - mixture_pdf(x, cal_laws, weights))
    rho = float(np.trapz(diff, x))
    return min(max(rho, 0.0), 2.0)


def robust_validation_quantities(
    cal_laws: List[ResidualLaw],
    weights: np.ndarray,
    alpha: float,
    rho_wc: float,
) -> dict[str, float]:
    """Compute standard and TV-RV quantiles/widths for a calibration mixture."""

    cal_cdf = lambda x: mixture_cdf(x, cal_laws, weights)
    upper = integration_upper_bound(cal_laws, multiplier=12.0)
    q_std = adaptive_quantile(cal_cdf, 1.0 - alpha, initial_upper=upper)
    u_rv = min(1.0 - alpha + rho_wc / 2.0, 1.0)
    q_rv = adaptive_quantile(cal_cdf, u_rv, initial_upper=upper)
    return {
        "q_std": q_std,
        "width_std": 2.0 * q_std,
        "rho_wc": float(rho_wc),
        "u_rv": u_rv,
        "q_rv": q_rv,
        "width_rv": float("inf") if np.isinf(q_rv) else 2.0 * q_rv,
    }


def score_coverage(q: float, law: ResidualLaw) -> float:
    """Coverage P(|U|<=q) for U from a residual law."""

    if np.isinf(q):
        return 1.0
    return float(folded_normal_cdf(q, law.mu, law.sigma))


def residual_law_from_row(row: dict[str, float]) -> ResidualLaw:
    """Convert a residual parameter row to ResidualLaw."""

    return ResidualLaw(row["name"], float(row["mu"]), float(row["sigma"]))
