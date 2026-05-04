import numpy as np

from option_gpr.grids import GridSet, make_random_price_grid
from option_gpr.hyperparams import (
    residual_tuning_objective,
    tune_rbf_kernel_residual,
)
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator
from option_gpr.payoffs import call_boundary_values_log


def _model() -> BlackScholesModel:
    return BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)


def _operator_factory(model: BlackScholesModel, kernel):
    return BSLogOperator(model=model, kernel=kernel)


def _grid(seed: int) -> GridSet:
    model = _model()
    return make_random_price_grid(
        n_int=8,
        n_terminal=6,
        n_lower=3,
        n_upper=3,
        t_min=0.0,
        t_max=0.9,
        maturity=model.maturity,
        S_min=20.0,
        S_max=200.0,
        boundary_value_fn=lambda X: call_boundary_values_log(
            X,
            strike=model.strike,
            maturity=model.maturity,
            r=model.r,
            S_min=20.0,
            S_max=200.0,
        ),
        seed=seed,
    )


def _objective(train_grid: GridSet, tune_grid: GridSet) -> float:
    return residual_tuning_objective(
        np.log(np.array([100.0, 0.8, 1.2])),
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=train_grid,
        tune_grid=tune_grid,
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
    )


def test_residual_tuning_objective_returns_finite_scalar() -> None:
    value = _objective(_grid(seed=1), _grid(seed=2))

    assert np.isscalar(value)
    assert np.isfinite(value)
    assert value >= 0.0


def test_residual_tuning_objective_uses_separate_tuning_boundary_values() -> None:
    train_grid = _grid(seed=1)
    tune_grid = _grid(seed=2)
    shifted_tune_grid = GridSet(
        X_int=tune_grid.X_int,
        X_bd=tune_grid.X_bd,
        y_bd=tune_grid.y_bd + 10.0,
    )

    original = _objective(train_grid, tune_grid)
    shifted = _objective(train_grid, shifted_tune_grid)

    assert shifted != original


def test_residual_tuning_objective_returns_penalty_for_nonfinite_theta() -> None:
    penalty = 12345.0

    value = residual_tuning_objective(
        np.array([np.nan, 0.0, 0.0]),
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=_grid(seed=1),
        tune_grid=_grid(seed=2),
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
        penalty=penalty,
    )

    assert value == penalty


def test_residual_tuning_objective_supports_fixed_sigma_f() -> None:
    value = residual_tuning_objective(
        np.log(np.array([0.8, 1.2])),
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=_grid(seed=1),
        tune_grid=_grid(seed=2),
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
        fixed_sigma_f=1.0,
    )

    assert np.isscalar(value)
    assert np.isfinite(value)
    assert value >= 0.0


def test_residual_tuning_objective_fixed_sigma_f_rejects_three_parameter_theta() -> None:
    penalty = 12345.0

    value = residual_tuning_objective(
        np.log(np.array([1.0, 0.8, 1.2])),
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=_grid(seed=1),
        tune_grid=_grid(seed=2),
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
        fixed_sigma_f=1.0,
        penalty=penalty,
    )

    assert value == penalty


def test_tune_rbf_kernel_residual_tiny_problem_returns_finite_parameters() -> None:
    result = tune_rbf_kernel_residual(
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=_grid(seed=1),
        tune_grid=_grid(seed=2),
        initial_sigma_f=100.0,
        initial_ell_t=0.8,
        initial_ell_x=1.2,
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
        maxiter=2,
    )

    assert result.sigma_f > 0.0
    assert result.ell_t > 0.0
    assert result.ell_x > 0.0
    assert np.all(np.isfinite(result.theta_log))
    assert np.isfinite(result.objective_value)
    assert result.nfev > 0


def test_tune_rbf_kernel_residual_fixed_sigma_f_optimizes_only_lengthscales() -> None:
    result = tune_rbf_kernel_residual(
        model=_model(),
        operator_factory=_operator_factory,
        train_grid=_grid(seed=1),
        tune_grid=_grid(seed=2),
        initial_sigma_f=100.0,
        initial_ell_t=0.8,
        initial_ell_x=1.2,
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
        maxiter=2,
        fixed_sigma_f=1.0,
    )

    assert result.sigma_f == 1.0
    assert result.ell_t > 0.0
    assert result.ell_x > 0.0
    assert result.theta_log.shape == (2,)
    assert np.all(np.isfinite(result.theta_log))
    assert np.isfinite(result.objective_value)
