"""Uncertainty-set objects and scalar worst-case reductions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .scm_core import SCMEnvironment, SCMSetting, second_moment_noise, sym
from .settings import CovOnlySetting, MeanOnlySetting


def S0(setting: SCMSetting) -> np.ndarray:
    """Reference second moment S_0."""

    return second_moment_noise(setting.observed_envs[0])


def S_gamma(setting: SCMSetting, gamma: float) -> np.ndarray:
    """DRIG second-moment matrix S_gamma = S0 + gamma sum w_e (S_e-S0)."""

    S_ref = S0(setting)
    diff = np.zeros_like(S_ref)
    for env, weight in zip(setting.observed_envs, setting.weights):
        diff += weight * (second_moment_noise(env) - S_ref)
    return sym(S_ref + gamma * diff)


def M_obs_mean(setting: MeanOnlySetting) -> np.ndarray:
    """M_obs = sum w_e delta^e delta^e^T for mean-only setting."""

    d = setting.p + 1
    M = np.zeros((d, d))
    for env, weight in zip(setting.observed_envs, setting.weights):
        M += weight * np.outer(env.delta, env.delta)
    return sym(M)


def Delta_obs_cov(setting: CovOnlySetting) -> np.ndarray:
    """Delta_obs = sum w_e H_e for covariance-only setting."""

    if setting.Omega0 is None:
        raise ValueError("CovOnlySetting must define Omega0")
    Delta = np.zeros_like(setting.Omega0)
    for env, weight in zip(setting.observed_envs, setting.weights):
        Delta += weight * (env.Omega - setting.Omega0)
    return sym(Delta)


def mean_only_m_max(w: np.ndarray, M_obs: np.ndarray, gamma: float) -> float:
    """Maximum absolute projected mean under delta delta^T <= gamma M_obs."""

    val = max(0.0, float(gamma * w.T @ M_obs @ w))
    return float(np.sqrt(val))


def cov_only_h_max(w: np.ndarray, Delta_obs: np.ndarray, gamma: float) -> float:
    """Maximum projected covariance perturbation under 0 <= H <= gamma Delta_obs."""

    return max(0.0, float(gamma * w.T @ Delta_obs @ w))


def gaussian_moment_halfwidth(w: np.ndarray, Sg: np.ndarray, alpha: float) -> float:
    """Half-width from the Gaussian second-moment interval.

    q = sqrt(w^T S_gamma w) * sqrt(1 + z_{1-alpha/2}^2).
    """

    M = max(0.0, float(w.T @ Sg @ w))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(np.sqrt(M) * np.sqrt(1.0 + z**2))


def is_psd(A: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True if symmetric part is positive semidefinite."""

    return bool(np.min(np.linalg.eigvalsh(sym(A))) >= -tol)
