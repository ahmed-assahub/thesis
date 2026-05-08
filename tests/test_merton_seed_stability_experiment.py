import csv
import json
from pathlib import Path

import numpy as np

from experiments.merton_seed_stability import (
    ExperimentConfig,
    make_grid,
    run_experiment,
    write_outputs,
)


def _tiny_config(
    output_dir: Path, grid_sampling: str = "price_uniform"
) -> ExperimentConfig:
    return ExperimentConfig(
        maturities=(0.5,),
        S_min=20.0,
        S_max=200.0,
        validation_S_min=90.0,
        validation_S_max=110.0,
        validation_S_step=10.0,
        n_seeds=2,
        n_int=9,
        n_bd=9,
        maxiter=2,
        tail_tol=1e-10,
        max_terms=50,
        output_dir=output_dir,
        grid_sampling=grid_sampling,
        show_progress=False,
    )


def test_run_experiment_returns_expected_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    seed_rows, summary_rows, maturity_rows, hyper_rows, prediction_rows = run_experiment(
        config
    )

    assert len(seed_rows) == 2
    assert len(summary_rows) == 11
    assert len(maturity_rows) == 2
    assert len(hyper_rows) == 2
    assert len(prediction_rows) == 6
    assert {
        "seed_index",
        "seed_base",
        "mae",
        "max_abs_error",
        "mre_itm",
        "mean_posterior_std",
        "optimization_time_sec",
        "final_fit_time_sec",
        "pricing_time_sec",
        "posterior_std_time_sec",
        "series_benchmark_time_sec",
        "total_gpr_price_time_sec",
        "total_gpr_with_std_time_sec",
    }.issubset(seed_rows[0])
    assert {
        "seed_index",
        "seed_base",
        "maturity",
        "sigma_f",
        "ell_t",
        "ell_x",
        "objective_value",
    }.issubset(hyper_rows[0])
    assert {
        "seed_index",
        "maturity",
        "n_int",
        "n_bd",
        "S0",
        "prediction",
        "reference_series",
        "spread",
        "abs_error",
        "posterior_std",
    }.issubset(prediction_rows[0])


def test_summary_metrics_include_requested_statistics(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    _, summary_rows, *_ = run_experiment(config)

    summary_by_metric = {row["metric"]: row for row in summary_rows}
    for metric in ("mae", "max_abs_error", "mre_itm"):
        row = summary_by_metric[metric]
        assert {"metric", "mean", "median", "std", "min", "max"} == set(row)
        for key in ("mean", "median", "std", "min", "max"):
            assert np.isfinite(row[key])


def test_metrics_hyperparameters_and_posterior_std_are_valid(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    seed_rows, _, maturity_rows, hyper_rows, prediction_rows = run_experiment(config)

    for row in seed_rows + maturity_rows:
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["max_abs_error"])
        assert np.isfinite(row["mre_itm"])
        assert np.isfinite(row["mean_posterior_std"])
        assert row["mae"] >= 0.0
        assert row["max_abs_error"] >= 0.0
        assert row["mre_itm"] >= 0.0
        assert row["mean_posterior_std"] >= 0.0
        assert row["max_posterior_std"] >= 0.0

    for row in hyper_rows:
        assert row["sigma_f"] > 0.0
        assert row["ell_t"] > 0.0
        assert row["ell_x"] > 0.0
        assert np.isfinite(row["sigma_f"])
        assert np.isfinite(row["ell_t"])
        assert np.isfinite(row["ell_x"])

    for row in prediction_rows:
        assert np.isfinite(row["reference_series"])
        assert np.isfinite(row["spread"])
        assert np.isfinite(row["abs_error"])
        assert np.isfinite(row["posterior_std"])
        assert row["abs_error"] >= 0.0
        assert row["posterior_std"] >= 0.0


def test_write_outputs_creates_expected_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    rows = run_experiment(config)

    write_outputs(config, *rows)

    expected_files = {
        "seed_metrics.csv",
        "summary_metrics.csv",
        "maturity_metrics.csv",
        "hyperparameters.csv",
        "predictions.csv",
        "config.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "summary_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with (tmp_path / "config.json").open(encoding="utf-8") as handle:
        config_data = json.load(handle)

    assert {"mae", "max_abs_error", "mre_itm"}.issubset(
        {row["metric"] for row in summary_rows}
    )
    assert {"metric", "mean", "median", "std", "min", "max"} == set(
        summary_rows[0]
    )
    assert "reference_series" in prediction_rows[0]
    assert "spread" in prediction_rows[0]
    assert config_data["grid_sampling"] == "price_uniform"
    assert "Merton semi-analytical series benchmark" in config_data[
        "error_reference_note"
    ]


def test_log_price_uniform_make_grid_uses_log_bounds(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, grid_sampling="log_price_uniform")

    grid = make_grid(config, maturity=0.5, seed=123)

    lower = np.log(config.S_min)
    upper = np.log(config.S_max)
    assert np.all(grid.X_int[:, 1] >= lower)
    assert np.all(grid.X_int[:, 1] <= upper)
    assert np.all(grid.X_bd[:, 1] >= lower)
    assert np.all(grid.X_bd[:, 1] <= upper)
