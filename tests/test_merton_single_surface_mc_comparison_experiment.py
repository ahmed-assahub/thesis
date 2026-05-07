import csv
import json
from pathlib import Path

import numpy as np

from experiments.merton_single_surface_mc_comparison import (
    ExperimentConfig,
    make_grid,
    run_experiment,
    write_outputs,
)


def _tiny_config(
    output_dir: Path, grid_sampling: str = "price_uniform"
) -> ExperimentConfig:
    return ExperimentConfig(
        maturities=(0.5, 1.0),
        S_min=20.0,
        S_max=200.0,
        validation_S_min=90.0,
        validation_S_max=110.0,
        validation_S_step=10.0,
        n_int=8,
        n_bd=8,
        mc_path_values=(100, 200),
        maxiter=2,
        tail_tol=1e-10,
        max_terms=50,
        output_dir=output_dir,
        grid_sampling=grid_sampling,
    )


def test_run_experiment_returns_expected_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    rows = run_experiment(config)
    (
        gpr_aggregate_rows,
        gpr_maturity_rows,
        gpr_hyperparameter_rows,
        gpr_prediction_rows,
        mc_aggregate_rows,
        mc_maturity_rows,
        mc_prediction_rows,
        series_reference_rows,
    ) = rows

    assert len(gpr_aggregate_rows) == 1
    assert len(gpr_maturity_rows) == 2
    assert len(gpr_hyperparameter_rows) == 1
    assert len(gpr_prediction_rows) == 6
    assert len(mc_aggregate_rows) == 2
    assert len(mc_maturity_rows) == 4
    assert len(mc_prediction_rows) == 12
    assert len(series_reference_rows) == 6

    assert {
        "remaining_maturity",
        "eval_t",
        "reference_series",
        "spread",
        "abs_error",
    }.issubset(gpr_prediction_rows[0])
    assert {
        "remaining_maturity",
        "reference_series",
        "spread",
        "abs_error",
    }.issubset(mc_prediction_rows[0])
    assert {
        "n_int",
        "n_bd",
        "total_gpr_price_time_sec",
        "total_gpr_with_std_time_sec",
    }.issubset(gpr_aggregate_rows[0])


def test_eval_times_match_remaining_maturities(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    _, gpr_maturity_rows, _, gpr_prediction_rows, *_ = run_experiment(config)

    maturity_to_eval_t = {
        row["remaining_maturity"]: row["eval_t"] for row in gpr_maturity_rows
    }
    assert maturity_to_eval_t[0.5] == 0.5
    assert maturity_to_eval_t[1.0] == 0.0

    prediction_eval_times = {
        row["remaining_maturity"]: row["eval_t"] for row in gpr_prediction_rows
    }
    assert prediction_eval_times[0.5] == 0.5
    assert prediction_eval_times[1.0] == 0.0


def test_metrics_and_hyperparameters_are_valid(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    (
        gpr_aggregate_rows,
        _,
        gpr_hyperparameter_rows,
        gpr_prediction_rows,
        mc_aggregate_rows,
        _,
        mc_prediction_rows,
        _,
    ) = run_experiment(config)

    for row in gpr_aggregate_rows + mc_aggregate_rows:
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["max_abs_error"])
        assert np.isfinite(row["mre_itm"])
        assert row["mae"] >= 0.0
        assert row["max_abs_error"] >= 0.0
        assert row["mre_itm"] >= 0.0

    for row in gpr_hyperparameter_rows:
        assert row["sigma_f"] > 0.0
        assert row["ell_t"] > 0.0
        assert row["ell_x"] > 0.0
        assert np.isfinite(row["sigma_f"])
        assert np.isfinite(row["ell_t"])
        assert np.isfinite(row["ell_x"])

    for row in gpr_prediction_rows + mc_prediction_rows:
        assert np.isfinite(row["reference_series"])
        assert np.isfinite(row["spread"])
        assert np.isfinite(row["abs_error"])
        assert row["abs_error"] >= 0.0


def test_write_outputs_creates_expected_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    rows = run_experiment(config)

    write_outputs(config, *rows)

    expected_files = {
        "gpr_aggregate_metrics.csv",
        "gpr_maturity_metrics.csv",
        "gpr_hyperparameters.csv",
        "gpr_predictions.csv",
        "mc_aggregate_metrics.csv",
        "mc_maturity_metrics.csv",
        "mc_predictions.csv",
        "series_reference.csv",
        "config.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "gpr_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gpr_rows = list(csv.DictReader(handle))
    with (tmp_path / "mc_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        mc_rows = list(csv.DictReader(handle))
    with (tmp_path / "config.json").open(encoding="utf-8") as handle:
        config_data = json.load(handle)

    assert len(gpr_rows) == 6
    assert len(mc_rows) == 12
    assert "reference_series" in gpr_rows[0]
    assert "reference_series" in mc_rows[0]
    assert config_data["surface_design"] == "single_gpr_surface_on_T_max"
    assert "Merton semi-analytical series benchmark" in config_data[
        "error_reference_note"
    ]
    assert config_data["grid_sampling"] == "price_uniform"
    assert "log_price_uniform" in config_data["grid_sampling_note"]


def test_log_price_uniform_make_grid_uses_log_bounds(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, grid_sampling="log_price_uniform")

    grid = make_grid(config, maturity=1.0, seed=123)

    lower = np.log(config.S_min)
    upper = np.log(config.S_max)
    assert np.all(grid.X_int[:, 1] >= lower)
    assert np.all(grid.X_int[:, 1] <= upper)
    assert np.all(grid.X_bd[:, 1] >= lower)
    assert np.all(grid.X_bd[:, 1] <= upper)


def test_log_price_uniform_run_experiment_returns_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, grid_sampling="log_price_uniform")

    rows = run_experiment(config)

    assert len(rows[0]) == 1
    assert len(rows[4]) == len(config.mc_path_values)
