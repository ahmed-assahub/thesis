import csv
import json
from pathlib import Path

import numpy as np

from experiments.black_scholes_collocation_sensitivity_nested import (
    ExperimentConfig,
    make_nested_grid_pool,
    run_experiment,
    split_boundary_count,
    trim_boundary_blocks,
    trim_nested_grid,
    write_outputs,
)


def _tiny_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        maturities=(0.5,),
        S_min=20.0,
        S_max=200.0,
        validation_S_min=90.0,
        validation_S_max=110.0,
        validation_S_step=10.0,
        n_int_values=(4, 8),
        n_bd_ratios=(0.75, 1.5),
        maxiter=2,
        output_dir=output_dir,
    )


def test_trim_nested_grid_returns_expected_shapes(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    pool = make_nested_grid_pool(
        config,
        maturity=0.5,
        max_n_int=8,
        max_n_bd=12,
        seed=1,
        boundary_counts=(4, 4, 4),
    )

    grid = trim_nested_grid(config, maturity=0.5, pool=pool, n_int=4, n_bd=6)

    assert grid.X_int.shape == (4, 2)
    assert grid.X_bd.shape == (6, 2)
    assert grid.y_bd.shape == (6,)


def test_nested_grid_uses_prefixes_for_interior_and_boundary_blocks(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    pool = make_nested_grid_pool(
        config,
        maturity=0.5,
        max_n_int=8,
        max_n_bd=12,
        seed=1,
        boundary_counts=(4, 4, 4),
    )

    small_grid = trim_nested_grid(config, maturity=0.5, pool=pool, n_int=4, n_bd=6)
    large_grid = trim_nested_grid(config, maturity=0.5, pool=pool, n_int=8, n_bd=12)
    small_terminal, small_lower, small_upper = split_boundary_count(6)
    large_blocks = trim_boundary_blocks(pool, 12)

    np.testing.assert_allclose(small_grid.X_int, large_grid.X_int[:4])
    np.testing.assert_allclose(
        small_grid.X_bd[:small_terminal],
        large_blocks.X_terminal[:small_terminal],
    )
    np.testing.assert_allclose(
        small_grid.X_bd[small_terminal : small_terminal + small_lower],
        large_blocks.X_lower[:small_lower],
    )
    np.testing.assert_allclose(
        small_grid.X_bd[-small_upper:],
        large_blocks.X_upper[:small_upper],
    )


def test_nested_grid_uses_prefixes_across_boundary_ratios(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    pool = make_nested_grid_pool(
        config,
        maturity=0.5,
        max_n_int=8,
        max_n_bd=12,
        seed=1,
        boundary_counts=(4, 4, 4),
    )

    small_blocks = trim_boundary_blocks(pool, n_bd=6)
    large_blocks = trim_boundary_blocks(pool, n_bd=12)

    np.testing.assert_allclose(
        small_blocks.X_terminal,
        large_blocks.X_terminal[: small_blocks.X_terminal.shape[0]],
    )
    np.testing.assert_allclose(
        small_blocks.X_lower,
        large_blocks.X_lower[: small_blocks.X_lower.shape[0]],
    )
    np.testing.assert_allclose(
        small_blocks.X_upper,
        large_blocks.X_upper[: small_blocks.X_upper.shape[0]],
    )


def test_run_experiment_returns_expected_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    aggregate_rows, maturity_rows, hyperparameter_rows = run_experiment(config)

    assert len(aggregate_rows) == 4
    assert len(maturity_rows) == 4
    assert len(hyperparameter_rows) == 4
    assert {
        "n_int",
        "n_bd",
        "mae",
        "max_abs_error",
        "mre_itm",
        "optimization_time_sec",
        "total_time_sec",
    }.issubset(aggregate_rows[0])
    assert {
        "maturity",
        "train_seed",
        "tune_seed",
        "tuning_objective_value",
    }.issubset(maturity_rows[0])
    assert {"sigma_f", "ell_t", "ell_x", "objective_value"}.issubset(
        hyperparameter_rows[0]
    )


def test_write_outputs_creates_expected_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    aggregate_rows, maturity_rows, hyperparameter_rows = run_experiment(config)

    write_outputs(config, aggregate_rows, maturity_rows, hyperparameter_rows)

    expected_files = {
        "aggregate_metrics.csv",
        "maturity_metrics.csv",
        "hyperparameters.csv",
        "config.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "aggregate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["n_int"] == "4"
    assert rows[0]["n_bd"] == "3"

    with (tmp_path / "config.json").open(encoding="utf-8") as handle:
        config_data = json.load(handle)

    assert "ratio_index" not in config_data["seed_policy"]
    assert "For each maturity" in config_data["grid_design_note"]
