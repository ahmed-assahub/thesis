import math

import pytest

from option_gpr.models import BlackScholesModel, MertonJumpDiffusionModel


def test_black_scholes_coefficients() -> None:
    model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)

    coeffs = model.coefficients()

    assert coeffs.a == pytest.approx(-0.05)
    assert coeffs.b == pytest.approx(0.05 - 0.5 * 0.2**2)
    assert coeffs.c == pytest.approx(0.5 * 0.2**2)


def test_merton_coefficients_reduce_to_black_scholes_when_jump_intensity_zero() -> None:
    bs_model = BlackScholesModel(r=0.05, sigma=0.2, strike=100.0, maturity=1.0)
    merton_model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.2,
        jump_intensity=0.0,
        jump_mean=-0.1,
        jump_std=0.3,
        strike=100.0,
        maturity=1.0,
    )

    assert merton_model.coefficients() == bs_model.coefficients()


def test_merton_coefficients_with_positive_jump_intensity() -> None:
    model = MertonJumpDiffusionModel(
        r=0.05,
        sigma=0.2,
        jump_intensity=0.4,
        jump_mean=-0.1,
        jump_std=0.3,
        strike=100.0,
        maturity=1.0,
    )

    coeffs = model.coefficients()
    jump_compensator = math.exp(-0.1 + 0.5 * 0.3**2) - 1.0

    assert coeffs.a == pytest.approx(-(0.05 + 0.4))
    assert coeffs.b == pytest.approx(0.05 - 0.5 * 0.2**2 - 0.4 * jump_compensator)
    assert coeffs.c == pytest.approx(0.5 * 0.2**2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0},
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": math.inf},
    ],
)
def test_black_scholes_model_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    params = {"r": 0.05, "sigma": 0.2, "strike": 100.0, "maturity": 1.0}
    params.update(kwargs)

    with pytest.raises(ValueError):
        BlackScholesModel(**params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0},
        {"jump_intensity": -0.1},
        {"jump_mean": math.inf},
        {"jump_std": 0.0},
        {"jump_std": -0.1},
        {"strike": 0.0},
        {"maturity": 0.0},
        {"r": math.inf},
    ],
)
def test_merton_model_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    params = {
        "r": 0.05,
        "sigma": 0.2,
        "jump_intensity": 0.4,
        "jump_mean": -0.1,
        "jump_std": 0.3,
        "strike": 100.0,
        "maturity": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        MertonJumpDiffusionModel(**params)
