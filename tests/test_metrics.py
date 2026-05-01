import numpy as np
import pytest

from option_gpr.metrics import mae, max_abs_error, mean_relative_error


def test_mae_matches_hand_computed_example() -> None:
    pred = np.array([10.0, 12.0, 14.0])
    ref = np.array([9.0, 12.5, 16.0])

    assert mae(pred, ref) == pytest.approx((1.0 + 0.5 + 2.0) / 3.0)


def test_max_abs_error_matches_hand_computed_example() -> None:
    pred = np.array([10.0, 12.0, 14.0])
    ref = np.array([9.0, 12.5, 16.0])

    assert max_abs_error(pred, ref) == pytest.approx(2.0)


def test_mean_relative_error_matches_hand_computed_example() -> None:
    pred = np.array([11.0, 18.0])
    ref = np.array([10.0, 20.0])

    assert mean_relative_error(pred, ref) == pytest.approx((0.1 + 0.1) / 2.0)


def test_mean_relative_error_uses_eps_for_zero_reference_values() -> None:
    pred = np.array([0.1, 0.2])
    ref = np.array([0.0, 1e-12])

    expected = np.mean(np.array([0.1, 0.2 - 1e-12]) / 1e-3)

    assert mean_relative_error(pred, ref, eps=1e-3) == pytest.approx(expected)


def test_mean_relative_error_supports_itm_masked_inputs() -> None:
    pred = np.array([0.0, 4.5, 11.0])
    ref = np.array([0.0, 5.0, 10.0])
    initial_prices = np.array([90.0, 105.0, 120.0])
    strike = 100.0
    itm = initial_prices > strike

    assert mean_relative_error(pred[itm], ref[itm]) == pytest.approx(
        ((0.5 / 5.0) + (1.0 / 10.0)) / 2.0
    )


@pytest.mark.parametrize(
    "pred, ref",
    [
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([[1.0, 2.0]])),
        (np.array([1.0]), np.array([1.0, 2.0])),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([1.0]), np.array([np.inf])),
    ],
)
def test_metrics_reject_invalid_inputs(pred: np.ndarray, ref: np.ndarray) -> None:
    with pytest.raises(ValueError):
        mae(pred, ref)

    with pytest.raises(ValueError):
        max_abs_error(pred, ref)

    with pytest.raises(ValueError):
        mean_relative_error(pred, ref)


@pytest.mark.parametrize("eps", [0.0, -1.0, np.inf, np.nan])
def test_mean_relative_error_rejects_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError):
        mean_relative_error(np.array([1.0]), np.array([1.0]), eps=eps)
