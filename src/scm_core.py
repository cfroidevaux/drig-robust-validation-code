"""Core fixed-B Gaussian SCM computations for uncertainty-set Chapter 4 runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SCMEnvironment:
    """Gaussian SCM noise law xi ~ N(delta, Omega)."""

    name: str
    delta: np.ndarray
    Omega: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", np.asarray(self.delta, dtype=float))
        object.__setattr__(self, "Omega", np.asarray(self.Omega, dtype=float))
        if self.delta.ndim != 1:
            raise ValueError(f"delta for {self.name} must be one-dimensional")
        if self.Omega.shape != (self.delta.size, self.delta.size):
            raise ValueError(f"Omega for {self.name} has incompatible shape")


@dataclass(frozen=True)
class SCMSetting:
    """Base class for fixed-B Gaussian SCM settings."""

    name: str
    p: int
    B: np.ndarray
    observed_envs: List[SCMEnvironment]
    weights: np.ndarray
    alpha: float = 0.1
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "B", np.asarray(self.B, dtype=float))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        d = self.p + 1
        if self.B.shape != (d, d):
            raise ValueError(f"B must have shape {(d, d)} for p={self.p}")
        if len(self.observed_envs) != len(self.weights):
            raise ValueError("Number of observed environments and weights must match")
        if not np.all(self.weights >= 0):
            raise ValueError("All weights must be nonnegative")
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("Weights must sum to one")
        for env in self.observed_envs:
            if env.delta.size != d:
                raise ValueError(f"Environment {env.name} has wrong dimension")
            if env.Omega.shape != (d, d):
                raise ValueError(f"Environment {env.name} has wrong covariance shape")


def sym(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return (A + A.T) / 2.0


def min_eigenvalue(A: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part."""

    return float(np.min(np.linalg.eigvalsh(sym(A))))


def is_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Return whether the symmetric part is positive definite up to tolerance."""

    return min_eigenvalue(A) > tol


def compute_K(B: np.ndarray) -> np.ndarray:
    """Compute K=(I-B)^(-1)."""

    d = B.shape[0]
    return np.linalg.inv(np.eye(d) - B)


def second_moment_noise(env: SCMEnvironment) -> np.ndarray:
    """Return S_e = E[xi xi^T] = Omega + delta delta^T."""

    return env.Omega + np.outer(env.delta, env.delta)


def compute_Lambda(K: np.ndarray, env: SCMEnvironment) -> np.ndarray:
    """Return Lambda_e = E[Z Z^T] for Z=K xi."""

    S = second_moment_noise(env)
    return K @ S @ K.T


def split_Lambda(Lambda: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return Lambda_XX, Lambda_XY, Lambda_YY."""

    return Lambda[:p, :p], Lambda[:p, p], float(Lambda[p, p])


def ell_vector(b: np.ndarray) -> np.ndarray:
    """Return ell_b=(-b,1)."""

    b = np.asarray(b, dtype=float)
    return np.concatenate([-b, np.array([1.0])])


def residual_direction(K: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return w_b = K^T ell_b."""

    return K.T @ ell_vector(b)


def risk_from_Lambda(Lambda: np.ndarray, b: np.ndarray) -> float:
    """Return R_e(b)=ell_b^T Lambda_e ell_b."""

    ell = ell_vector(b)
    return float(ell.T @ Lambda @ ell)


def residual_params(K: np.ndarray, b: np.ndarray, env: SCMEnvironment) -> Tuple[float, float]:
    """Return residual mean and variance in one environment."""

    w = residual_direction(K, b)
    mu = float(w.T @ env.delta)
    var = float(w.T @ env.Omega @ w)
    if var <= 0:
        raise ValueError(f"Residual variance for {env.name} is nonpositive: {var}")
    return mu, var


def weighted_observed_moments(
    K: np.ndarray, p: int, envs: List[SCMEnvironment], weights: np.ndarray
) -> Dict[str, np.ndarray | float]:
    """Return weighted observed Lambda blocks."""

    Lxx = np.zeros((p, p))
    Lxy = np.zeros(p)
    Lyy = 0.0
    for env, weight in zip(envs, weights):
        Lam = compute_Lambda(K, env)
        xx, xy, yy = split_Lambda(Lam, p)
        Lxx += weight * xx
        Lxy += weight * xy
        Lyy += weight * yy
    return {"xx": Lxx, "xy": Lxy, "yy": Lyy}


def reference_moments(K: np.ndarray, p: int, ref_env: SCMEnvironment) -> Dict[str, np.ndarray | float]:
    """Return Lambda blocks for the reference environment."""

    Lam = compute_Lambda(K, ref_env)
    xx, xy, yy = split_Lambda(Lam, p)
    return {"xx": xx, "xy": xy, "yy": yy}


def objective_matrices(
    gamma: float,
    reference: Dict[str, np.ndarray | float],
    observed: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return A_gamma, h_gamma, c_gamma for fixed-reference objective."""

    A = reference["xx"] + gamma * (observed["xx"] - reference["xx"])
    h = reference["xy"] + gamma * (observed["xy"] - reference["xy"])
    c = float(reference["yy"] + gamma * (observed["yy"] - reference["yy"]))
    return np.asarray(A), np.asarray(h), c


def solve_b_gamma(setting: SCMSetting, gamma: float) -> Dict[str, object]:
    """Compute b_gamma and related objects for one gamma."""

    K = compute_K(setting.B)
    ref = reference_moments(K, setting.p, setting.observed_envs[0])
    obs = weighted_observed_moments(K, setting.p, setting.observed_envs, setting.weights)
    A, h, c = objective_matrices(gamma, ref, obs)
    pd = is_positive_definite(A)
    result: Dict[str, object] = {
        "gamma": float(gamma),
        "K": K,
        "A": A,
        "h": h,
        "c": c,
        "A_min_eig": min_eigenvalue(A),
        "A_positive_definite": bool(pd),
    }
    if not pd:
        return result
    b = np.linalg.solve(A, h)
    w = residual_direction(K, b)
    result.update({"b": b, "w": w})
    return result


def observed_residual_laws(setting: SCMSetting, K: np.ndarray, b: np.ndarray) -> List[Dict[str, float]]:
    """Return residual parameters and risks for the observed environments."""

    rows = []
    for env in setting.observed_envs:
        mu, var = residual_params(K, b, env)
        Lam = compute_Lambda(K, env)
        rows.append(
            {
                "name": env.name,
                "mu": mu,
                "var": var,
                "sigma": float(np.sqrt(var)),
                "risk": risk_from_Lambda(Lam, b),
            }
        )
    return rows
