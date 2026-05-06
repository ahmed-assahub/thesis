import numpy as np

from option_gpr.grids import make_random_price_grid
from option_gpr.kernels import RBFKernel
from option_gpr.models import BlackScholesModel, MertonJumpDiffusionModel
from option_gpr.operators import BSLogOperator, MJDOperator
from option_gpr.payoffs import call_boundary_values_log
from option_gpr.posterior import StackedOperatorGP


def _make_mjd_operator(jump_intensity: float = 0.4) -> MJDOperator:
    model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.2,
        jump_intensity=jump_intensity,
        jump_mean=-0.1,
        jump_std=0.3,
        strike=100.0,
        maturity=1.0,
    )
    kernel = RBFKernel(ell_t=0.7, ell_x=1.3, sigma_f=1.5)
    return MJDOperator(model=model, kernel=kernel)


def _sample_points() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 3.9], [0.4, 4.0], [0.8, 4.2]])
    Y = np.array([[0.2, 3.8], [0.6, 4.1]])
    return X, Y


def test_mjd_operator_block_shapes() -> None:
    operator = _make_mjd_operator()
    X, Y = _sample_points()

    assert operator.Lk(X, Y).shape == (3, 2)
    assert operator.kLp(X, Y).shape == (3, 2)
    assert operator.LkLp(X, Y).shape == (3, 2)


def test_mjd_operator_outputs_are_finite_with_positive_jump_intensity() -> None:
    operator = _make_mjd_operator(jump_intensity=0.4)
    X, Y = _sample_points()

    assert np.all(np.isfinite(operator.Lk(X, Y)))
    assert np.all(np.isfinite(operator.kLp(X, Y)))
    assert np.all(np.isfinite(operator.LkLp(X, Y)))


def test_mjd_operator_reduces_to_black_scholes_when_jump_intensity_zero() -> None:
    mjd_model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.2,
        jump_intensity=0.0,
        jump_mean=-0.1,
        jump_std=0.3,
        strike=100.0,
        maturity=1.0,
    )
    bs_model = BlackScholesModel(
        r=mjd_model.r,
        sigma=mjd_model.sigma,
        strike=mjd_model.strike,
        maturity=mjd_model.maturity,
    )
    kernel = RBFKernel(ell_t=0.7, ell_x=1.3, sigma_f=1.5)
    mjd_operator = MJDOperator(model=mjd_model, kernel=kernel)
    bs_operator = BSLogOperator(model=bs_model, kernel=kernel)
    X, Y = _sample_points()

    np.testing.assert_allclose(mjd_operator.Lk(X, Y), bs_operator.Lk(X, Y))
    np.testing.assert_allclose(mjd_operator.kLp(X, Y), bs_operator.kLp(X, Y))
    np.testing.assert_allclose(mjd_operator.LkLp(X, Y), bs_operator.LkLp(X, Y))


def test_mjd_operator_first_and_second_argument_symmetry() -> None:
    operator = _make_mjd_operator(jump_intensity=0.4)
    X, Y = _sample_points()

    np.testing.assert_allclose(operator.Lk(X, Y), operator.kLp(Y, X).T)


def test_mjd_operator_LkLp_symmetry() -> None:
    operator = _make_mjd_operator(jump_intensity=0.4)
    X, Y = _sample_points()

    np.testing.assert_allclose(operator.LkLp(X, Y), operator.LkLp(Y, X).T)


def test_mjd_operator_tiny_stacked_gp_pipeline() -> None:
    r = 0.05
    strike = 100.0
    maturity = 1.0
    S_min = 20.0
    S_max = 200.0
    model = MertonJumpDiffusionModel(
        r=r,
        sigma=0.2,
        jump_intensity=0.25,
        jump_mean=-0.1,
        jump_std=0.3,
        strike=strike,
        maturity=maturity,
    )
    kernel = RBFKernel(ell_t=0.8, ell_x=1.2, sigma_f=20.0)
    operator = MJDOperator(model=model, kernel=kernel)
    grid = make_random_price_grid(
        n_int=8,
        n_terminal=6,
        n_lower=3,
        n_upper=3,
        t_min=0.0,
        t_max=0.95,
        maturity=maturity,
        S_min=S_min,
        S_max=S_max,
        boundary_value_fn=lambda X: call_boundary_values_log(
            X,
            strike=strike,
            maturity=maturity,
            r=r,
            S_min=S_min,
            S_max=S_max,
        ),
        seed=123,
    )
    gp = StackedOperatorGP(
        model=model,
        kernel=kernel,
        operator=operator,
        noise_int=1e-2,
        noise_bd=1e-2,
        jitter=1e-6,
    )
    gp.fit(grid.X_int, grid.X_bd, grid.y_bd)

    S0 = np.array([80.0, 100.0, 120.0])
    X_star = np.column_stack([np.zeros_like(S0), np.log(S0)])
    pred, var = gp.predict(X_star, return_var=True)

    assert pred.shape == (3,)
    assert var.shape == (3,)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(var))
