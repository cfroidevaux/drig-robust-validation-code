"""Settings for the Chapter 4 uncertainty-set population study.

The examples are designed for the revised Chapter 4 story:

1. ``mean_only_p2`` validates the mean-only TV-control mechanism.
2. ``cov_only_p2`` validates the covariance-only TV-control mechanism.
3. ``mixed_second_moment_p2`` illustrates the full mixed second-moment case:
   TV-RV over the full set is vacuous, but Gaussian second-moment intervals
   remain finite and can be compared on a fixed target uncertainty set.

All examples use p=2 covariates.  This avoids making the numerical study look
like a one-dimensional accident while keeping the population calculations small
and transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .scm_core import SCMEnvironment, SCMSetting, sym


@dataclass(frozen=True)
class MeanOnlySetting(SCMSetting):
    """Observed environments share Omega and differ only in delta."""

    Omega: np.ndarray | None = None
    target_gamma: float = 5.0


@dataclass(frozen=True)
class CovOnlySetting(SCMSetting):
    """Observed environments share delta and have Omega_e = Omega0 + H_e."""

    delta_common: np.ndarray | None = None
    Omega0: np.ndarray | None = None
    target_gamma: float = 5.0


@dataclass(frozen=True)
class MixedSetting(SCMSetting):
    """General Gaussian setting for Gaussian second-moment intervals."""

    target_gamma: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def B_explicit(beta: np.ndarray) -> np.ndarray:
    """Fixed B for X=xi_X, Y=beta^T X + xi_Y.

    The noise vector is xi=(xi_X1,...,xi_Xp,xi_Y).  The structural matrix has
    beta in the last row and zero elsewhere, so K=(I-B)^{-1} gives
    Y=beta^T X+xi_Y.
    """

    beta = np.asarray(beta, dtype=float)
    p = beta.size
    B = np.zeros((p + 1, p + 1), dtype=float)
    B[p, :p] = beta
    return B


def make_env(name: str, delta: np.ndarray, Omega: np.ndarray) -> SCMEnvironment:
    """Create an environment after symmetrizing Omega."""

    return SCMEnvironment(
        name=name,
        delta=np.asarray(delta, dtype=float),
        Omega=sym(np.asarray(Omega, dtype=float)),
    )


def _assert_pd(Omega: np.ndarray, name: str) -> None:
    vals = np.linalg.eigvalsh(sym(Omega))
    if np.min(vals) <= 1e-10:
        raise ValueError(f"{name} is not positive definite; min eig={np.min(vals)}")


def _weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=float)


# ---------------------------------------------------------------------------
# Setting A: mean-only structured Gaussian subclass, p=2
# ---------------------------------------------------------------------------


def setting_mean_only_p2() -> MeanOnlySetting:
    """Mean-only Gaussian subclass with p=2.

    The observed mean shifts lie in the direction u=(1,1,1).  For the explicit
    SCM, w_b=(beta_1-b_1,beta_2-b_2,1), so the constraint w_b^T u=0 is feasible.
    The DRIG path can therefore learn a residual direction that is insensitive
    to the observed mean-shift direction.
    """

    beta = np.array([2.0, -0.75], dtype=float)
    B = B_explicit(beta)
    p = beta.size
    Omega = np.eye(p + 1)

    u = np.array([1.0, 1.0, 1.0], dtype=float)
    amplitudes = [0.0, 0.75, 1.50]
    obs = [make_env(f"obs_mean_{i}", a * u, Omega) for i, a in enumerate(amplitudes)]
    weights = _weights(len(obs))
    _assert_pd(Omega, "Omega")

    return MeanOnlySetting(
        name="mean_only_p2",
        p=p,
        B=B,
        observed_envs=obs,
        weights=weights,
        alpha=0.1,
        description=(
            "p=2 mean-only Gaussian subclass. The DRIG penalty controls projected "
            "residual means over delta delta^T <= gamma M_obs."
        ),
        Omega=Omega,
        target_gamma=5.0,
    )


# ---------------------------------------------------------------------------
# Setting B: covariance-only structured Gaussian subclass, p=2
# ---------------------------------------------------------------------------


def setting_cov_only_p2() -> CovOnlySetting:
    """Covariance-only Gaussian subclass with PSD perturbations, p=2.

    Covariance perturbations are rank-one in direction u=(1,1,1), hence PSD.
    The invariant projected-variance condition w_b^T u=0 is feasible because
    w_b has free covariate components and fixed last component one.
    """

    beta = np.array([2.0, -0.75], dtype=float)
    B = B_explicit(beta)
    p = beta.size
    delta = np.array([0.55, 0.15, 0.55], dtype=float)
    Omega0 = np.eye(p + 1)

    u = np.array([1.0, 1.0, 1.0], dtype=float)
    H_base = np.outer(u, u)
    scales = [0.0, 1.5, 4.0]
    obs = [make_env(f"obs_cov_{i}", delta, Omega0 + s * H_base) for i, s in enumerate(scales)]
    weights = _weights(len(obs))
    for env in obs:
        _assert_pd(env.Omega, env.name)

    return CovOnlySetting(
        name="cov_only_p2",
        p=p,
        B=B,
        observed_envs=obs,
        weights=weights,
        alpha=0.1,
        description=(
            "p=2 covariance-only Gaussian subclass. The DRIG penalty controls "
            "projected residual variances over 0 <= H <= gamma Delta_obs."
        ),
        delta_common=delta,
        Omega0=Omega0,
        target_gamma=5.0,
    )


# ---------------------------------------------------------------------------
# Setting C: mixed Gaussian second-moment interval, p=2
# ---------------------------------------------------------------------------


def setting_mixed_second_moment_p2() -> MixedSetting:
    """Mixed mean and covariance heterogeneity for Gaussian moment intervals.

    This setting is not used for TV-RV over the full second-moment set.  The TV
    worst case over that set is maximal by Proposition 3.6.  Instead, the code
    evaluates Gaussian second-moment intervals.  To make gamma-comparisons
    meaningful, it reports both the interval for the gamma-dependent DRIG set
    and an interval for a fixed target set S_target=S_0+eta*Delta_S.
    """

    beta = np.array([2.0, -0.75], dtype=float)
    B = B_explicit(beta)
    p = beta.size
    Omega0 = np.eye(p + 1)
    u = np.array([1.0, 1.0, 1.0], dtype=float)
    H_base = np.outer(u, u)

    specs = [
        ("obs_ref", 0.0, 0.00),
        ("obs_mix_1", 0.75, 1.50),
        ("obs_mix_2", 1.50, 4.00),
    ]
    obs = []
    for name, a, hscale in specs:
        delta = a * u
        Omega = Omega0 + hscale * H_base
        obs.append(make_env(name, delta, Omega))
    weights = _weights(len(obs))
    for env in obs:
        _assert_pd(env.Omega, env.name)

    return MixedSetting(
        name="mixed_second_moment_p2",
        p=p,
        B=B,
        observed_envs=obs,
        weights=weights,
        alpha=0.1,
        description=(
            "p=2 mixed mean and covariance heterogeneity. Full-set TV-RV is "
            "vacuous, while Gaussian second-moment intervals remain finite."
        ),
        target_gamma=5.0,
    )


SETTINGS: Dict[str, SCMSetting] = {
    "mean_only_p2": setting_mean_only_p2(),
    "cov_only_p2": setting_cov_only_p2(),
    "mixed_second_moment_p2": setting_mixed_second_moment_p2(),
}
