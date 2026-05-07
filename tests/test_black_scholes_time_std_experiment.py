import csv
from pathlib import Path

import numpy as np

from experiments.black_scholes_time_std import (
    ExperimentConfig,
    make_nested_grid_pool,
    run_experiment,
    trim_nested_grid,
    write_outputs,
)
from experiments.black_scholes_collocation_sensitivity_nested import (
    ExperimentConfig as NestedExperimentConfig,
    make_nested_grid_pool as make_collocation_nested_grid_pool,
    trim_nested_grid as trim_collocation_nested_grid,
)


def _tiny_config(
    output_dir: Path,
    m_values: tuple[int, ...] = (8,),
    grid_sampling: str = "price_uniform",
) -> ExperimentConfig:
    return ExperimentConfig(
        maturities=(0.5,),
        S_min=20.0,
        S_max=200.0,
        validation_S_min=90.0,
        validation_S_max=110.0,
        validation_S_step=10.0,
        m_values=m_values,
        maxiter=2,
        output_dir=output_dir,
        grid_sampling=grid_sampling,
    )


def test_run_experiment_returns_expected_rows_and_std(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, m_values=(8,))

    aggregate_rows, maturity_rows, hyperparameter_rows, prediction_rows = run_experiment(
        config
    )

    assert len(aggregate_rows) == 1
    assert len(maturity_rows) == 1
    assert len(hyperparameter_rows) == 1
    assert len(prediction_rows) == 3
    assert {
        "m",
        "d",
        "mae",
        "max_abs_error",
        "mre_itm",
        "mean_posterior_std",
        "max_posterior_std",
        "benchmark_time_sec",
        "total_time_sec",
    }.issubset(aggregate_rows[0])
    assert {"sigma_f", "ell_t", "ell_x", "fixed_sigma_f"}.issubset(
        hyperparameter_rows[0]
    )
    assert {
        "maturity",
        "S0",
        "x0",
        "prediction",
        "reference",
        "abs_error",
        "is_itm",
        "posterior_std",
    }.issubset(prediction_rows[0])

    posterior_stds = np.array([row["posterior_std"] for row in prediction_rows])
    assert np.all(np.isfinite(posterior_stds))
    assert np.all(posterior_stds >= 0.0)
    assert maturity_rows[0]["benchmark_time_sec"] >= 0.0


def test_total_time_includes_all_timed_components(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, m_values=(8,))

    _, maturity_rows, _, _ = run_experiment(config)
    row = maturity_rows[0]
    expected = (
        row["optimization_time_sec"]
        + row["final_fit_time_sec"]
        + row["pricing_time_sec"]
        + row["posterior_std_time_sec"]
        + row["benchmark_time_sec"]
    )

    assert np.isclose(row["total_time_sec"], expected)


def test_write_outputs_creates_expected_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, m_values=(8,))
    rows = run_experiment(config)

    write_outputs(config, *rows)

    expected_files = {
        "aggregate_metrics.csv",
        "maturity_metrics.csv",
        "hyperparameters.csv",
        "predictions.csv",
        "config.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))

    assert len(prediction_rows) == 3
    assert prediction_rows[0]["m"] == "8"
    assert prediction_rows[0]["d"] == "8"


def test_log_price_uniform_grid_pool_uses_log_bounds(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, m_values=(8,), grid_sampling="log_price_uniform")

    pool = make_nested_grid_pool(
        config,
        maturity=0.5,
        max_m=max(config.m_values),
        root_seed=123,
    )
    grid = trim_nested_grid(config, maturity=0.5, pool=pool, m=8)

    lower = np.log(config.S_min)
    upper = np.log(config.S_max)
    assert np.all(grid.X_int[:, 1] >= lower)
    assert np.all(grid.X_int[:, 1] <= upper)
    assert np.all(grid.X_bd[:, 1] >= lower)
    assert np.all(grid.X_bd[:, 1] <= upper)


def test_grid_matches_collocation_nested_grid_for_ratio_one(
    tmp_path: Path,
) -> None:
    maturity = 0.5
    m = 8
    time_std_config = _tiny_config(tmp_path / "time_std", m_values=(m,))
    nested_config = NestedExperimentConfig(
        maturities=(maturity,),
        S_min=20.0,
        S_max=200.0,
        validation_S_min=90.0,
        validation_S_max=110.0,
        validation_S_step=10.0,
        n_int_values=(m,),
        n_bd_ratios=(1.0,),
        maxiter=2,
        output_dir=tmp_path / "nested",
    )
    train_seed = time_std_config.base_seed + 1

    time_std_pool = make_nested_grid_pool(
        time_std_config,
        maturity=maturity,
        max_m=max(time_std_config.m_values),
        root_seed=train_seed,
    )
    nested_pool = make_collocation_nested_grid_pool(
        nested_config,
        maturity=maturity,
        max_n_int=max(nested_config.n_int_values),
        max_n_bd=m,
        seed=train_seed,
    )

    time_std_grid = trim_nested_grid(time_std_config, maturity, time_std_pool, m=m)
    nested_grid = trim_collocation_nested_grid(
        nested_config,
        maturity=maturity,
        pool=nested_pool,
        n_int=m,
        n_bd=m,
    )

    np.testing.assert_allclose(time_std_grid.X_int, nested_grid.X_int)
    np.testing.assert_allclose(time_std_grid.X_bd, nested_grid.X_bd)
    np.testing.assert_allclose(time_std_grid.y_bd, nested_grid.y_bd)
