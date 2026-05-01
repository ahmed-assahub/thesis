import numpy as np

from option_gpr.benchmarks import black_scholes_call_price_log
from option_gpr.grids import make_random_price_grid
from option_gpr.kernels import RBFKernel
from option_gpr.metrics import mae
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator
from option_gpr.payoffs import call_boundary_values_log
from option_gpr.posterior import StackedOperatorGP


def test_black_scholes_pipeline_smoke() -> None:
    r = 0.05
    sigma = 0.2
    strike = 100.0
    maturity = 1.0
    S_min = 20.0
    S_max = 200.0

    model = BlackScholesModel(
        r=r,
        sigma=sigma,
        strike=strike,
        maturity=maturity,
    )
    kernel = RBFKernel(ell_t=0.8, ell_x=1.2, sigma_f=100.0)
    operator = BSLogOperator(model=model, kernel=kernel)
    gp = StackedOperatorGP(
        model=model,
        kernel=kernel,
        operator=operator,
        noise_int=1e-3,
        noise_bd=1e-3,
        jitter=1e-8,
    )
    grid = make_random_price_grid(
        n_int=30,
        n_terminal=24,
        n_lower=8,
        n_upper=8,
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

    gp.fit(grid.X_int, grid.X_bd, grid.y_bd)
    S0 = np.array([80.0, 100.0, 120.0])
    X_star = np.column_stack([np.zeros_like(S0), np.log(S0)])

    pred, cov = gp.predict(X_star, return_cov=True)
    ref = black_scholes_call_price_log(
        X_star,
        strike=strike,
        maturity=maturity,
        r=r,
        sigma=sigma,
    )
    error = mae(pred, ref)

    assert pred.shape == (3,)
    assert ref.shape == (3,)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(ref))
    assert np.isfinite(error)
    assert error < 10.0
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > -1e-8)
    np.testing.assert_allclose(cov, cov.T)
