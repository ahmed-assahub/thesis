"""Residual-based hyperparameter tuning for stacked-operator GPs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from option_gpr.grids import GridSet
from option_gpr.kernels import RBFKernel
from option_gpr.posterior import StackedOperatorGP


@dataclass(frozen=True)
class ResidualTuningResult:
    """Result of residual-based RBF kernel tuning."""

    sigma_f: float
    ell_t: float
    ell_x: float
    theta_log: NDArray[np.float64]
    objective_value: float
    success: bool
    message: str
    nit: int
    nfev: int


OperatorFactory = Callable[[Any, RBFKernel], Any]


def residual_tuning_objective(
    theta_log: ArrayLike,
    *,
    model: Any,
    operator_factory: OperatorFactory,
    train_grid: GridSet,
    tune_grid: GridSet,
    noise_int: float,
    noise_bd: float,
    jitter: float,
    penalty: float = 1e30,
) -> float:
    """Return PDE-residual plus boundary-residual tuning error."""

    parsed = _parse_theta_log(theta_log)
    if parsed is None:
        return penalty
    sigma_f, ell_t, ell_x = parsed

    try:
        kernel = RBFKernel(ell_t=ell_t, ell_x=ell_x, sigma_f=sigma_f)
        operator = operator_factory(model, kernel)
        gp = StackedOperatorGP(
            model=model,
            kernel=kernel,
            operator=operator,
            noise_int=noise_int,
            noise_bd=noise_bd,
            jitter=jitter,
        )
        gp.fit(train_grid.X_int, train_grid.X_bd, train_grid.y_bd)
        operator_residual = gp.predict_operator(tune_grid.X_int)
        boundary_residual = gp.predict(tune_grid.X_bd) - tune_grid.y_bd
        if not np.all(np.isfinite(operator_residual)) or not np.all(
            np.isfinite(boundary_residual)
        ):
            return penalty
        err_int = np.mean(operator_residual**2)
        err_bd = np.mean(boundary_residual**2)
        objective = float(err_int + err_bd)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return penalty

    if not np.isfinite(objective):
        return penalty
    return objective


def tune_rbf_kernel_residual(
    *,
    model: Any,
    operator_factory: OperatorFactory,
    train_grid: GridSet,
    tune_grid: GridSet,
    initial_sigma_f: float,
    initial_ell_t: float,
    initial_ell_x: float,
    noise_int: float,
    noise_bd: float,
    jitter: float,
    maxiter: int = 100,
    xatol: float = 1e-3,
    fatol: float = 1e-6,
    penalty: float = 1e30,
) -> ResidualTuningResult:
    """Tune RBF hyperparameters by residual minimization in log-space."""

    initial = np.array([initial_sigma_f, initial_ell_t, initial_ell_x], dtype=float)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError(
            "initial_sigma_f, initial_ell_t, and initial_ell_x must be "
            "positive and finite."
        )

    theta0 = np.log(initial)
    objective = partial(
        residual_tuning_objective,
        model=model,
        operator_factory=operator_factory,
        train_grid=train_grid,
        tune_grid=tune_grid,
        noise_int=noise_int,
        noise_bd=noise_bd,
        jitter=jitter,
        penalty=penalty,
    )
    result = minimize(
        objective,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol},
    )
    sigma_f, ell_t, ell_x = _exp_theta_or_penalty_values(result.x)
    objective_value = float(result.fun) if np.isfinite(result.fun) else float(penalty)
    return ResidualTuningResult(
        sigma_f=sigma_f,
        ell_t=ell_t,
        ell_x=ell_x,
        theta_log=np.asarray(result.x, dtype=float),
        objective_value=objective_value,
        success=bool(result.success),
        message=str(result.message),
        nit=int(result.nit),
        nfev=int(result.nfev),
    )


def _parse_theta_log(theta_log: ArrayLike) -> tuple[float, float, float] | None:
    theta = np.asarray(theta_log, dtype=float)
    if theta.shape != (3,) or not np.all(np.isfinite(theta)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        sigma_f, ell_t, ell_x = np.exp(theta)
    values = np.array([sigma_f, ell_t, ell_x], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return None
    return float(sigma_f), float(ell_t), float(ell_x)


def _exp_theta_or_penalty_values(theta_log: ArrayLike) -> tuple[float, float, float]:
    parsed = _parse_theta_log(theta_log)
    if parsed is None:
        return float("nan"), float("nan"), float("nan")
    return parsed
