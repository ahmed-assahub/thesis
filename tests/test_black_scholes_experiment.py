import csv
from pathlib import Path

import numpy as np

from experiments.black_scholes_collocation_sensitivity import (
    ExperimentConfig,
    run_experiment,
    run_single_case,
    split_boundary_count,
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
        n_int_values=(8,),
        n_bd_ratios=(1.125,),
        maxiter=2,
        output_dir=output_dir,
    )


def test_split_boundary_count_assigns_remainder_to_terminal() -> None:
    assert split_boundary_count(10) == (4, 3, 3)


def test_run_single_case_returns_finite_metrics_and_hyperparameters(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    maturity_row, hyperparameter_row, _, arrays = run_single_case(
        config,
        maturity=0.5,
        n_int=8,
        n_bd=9,
        maturity_index=0,
    )

    assert np.isfinite(maturity_row["mae"])
    assert np.isfinite(maturity_row["max_abs_error"])
    assert np.isfinite(maturity_row["mre_itm"])
    assert hyperparameter_row["sigma_f"] > 0.0
    assert hyperparameter_row["sigma_f"] == 1.0
    assert hyperparameter_row["fixed_sigma_f"] == 1.0
    assert hyperparameter_row["log_sigma_f"] == ""
    assert hyperparameter_row["ell_t"] > 0.0
    assert hyperparameter_row["ell_x"] > 0.0
    assert arrays["pred"].shape == arrays["ref"].shape


def test_run_experiment_returns_expected_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    aggregate_rows, maturity_rows, hyperparameter_rows = run_experiment(config)

    assert len(aggregate_rows) == 1
    assert len(maturity_rows) == 1
    assert len(hyperparameter_rows) == 1
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

    assert len(rows) == 1
    assert rows[0]["n_int"] == "8"
    assert rows[0]["n_bd"] == "9"

    with (tmp_path / "hyperparameters.csv").open(newline="", encoding="utf-8") as handle:
        hyper_rows = list(csv.DictReader(handle))

    assert hyper_rows[0]["sigma_f"] == "1.0"
    assert hyper_rows[0]["fixed_sigma_f"] == "1.0"
    assert hyper_rows[0]["log_sigma_f"] == ""
