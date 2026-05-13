import csv
import json
from pathlib import Path

import numpy as np

from experiments.merton_greeks import (
    ExperimentConfig,
    run_experiment,
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
        n_int=8,
        n_bd=9,
        maxiter=2,
        tail_tol=1e-10,
        max_terms=50,
        output_dir=output_dir,
    )


def test_run_experiment_returns_expected_rows(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    aggregate_rows, maturity_rows, prediction_rows, hyperparameter_rows = run_experiment(
        config
    )

    assert len(aggregate_rows) == 1
    assert len(maturity_rows) == 1
    assert len(prediction_rows) == 3
    assert len(hyperparameter_rows) == 1
    assert {
        "price_mae",
        "price_max_abs_error",
        "price_mre_itm",
        "delta_mae",
        "delta_max_abs_error",
        "gamma_mae",
        "gamma_max_abs_error",
        "theta_mae",
        "theta_max_abs_error",
        "greek_time_sec",
    }.issubset(aggregate_rows[0])
    assert {
        "price_reference",
        "price_error",
        "price_abs_error",
        "delta_reference",
        "delta_error",
        "delta_abs_error",
        "gamma_reference",
        "gamma_error",
        "gamma_abs_error",
        "theta_reference",
        "theta_error",
        "theta_abs_error",
    }.issubset(prediction_rows[0])
    assert {
        "delta_mean_abs_reference",
        "delta_rel_mae",
        "gamma_mean_abs_reference",
        "gamma_rel_mae",
        "theta_mean_abs_reference",
        "theta_rel_mae",
    }.issubset(maturity_rows[0])


def test_metrics_and_hyperparameters_are_valid(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)

    aggregate_rows, maturity_rows, prediction_rows, hyperparameter_rows = run_experiment(
        config
    )

    metric_names = [
        "price_mae",
        "price_max_abs_error",
        "price_mre_itm",
        "delta_mae",
        "delta_max_abs_error",
        "gamma_mae",
        "gamma_max_abs_error",
        "theta_mae",
        "theta_max_abs_error",
    ]
    for row in aggregate_rows + maturity_rows:
        for name in metric_names:
            assert np.isfinite(row[name])
            assert row[name] >= 0.0

    for row in maturity_rows:
        for name in ("delta", "gamma", "theta"):
            mean_abs_reference = row[f"{name}_mean_abs_reference"]
            rel_mae = row[f"{name}_rel_mae"]
            assert np.isfinite(mean_abs_reference)
            assert np.isfinite(rel_mae)
            assert mean_abs_reference > 0.0
            assert rel_mae >= 0.0
            np.testing.assert_allclose(
                rel_mae,
                row[f"{name}_mae"] / mean_abs_reference,
            )

    for row in prediction_rows:
        for name in ("price", "delta", "gamma", "theta"):
            assert np.isfinite(row[f"{name}_reference"])
            assert np.isfinite(row[f"{name}_error"])
            assert np.isfinite(row[f"{name}_abs_error"])
            assert row[f"{name}_abs_error"] >= 0.0

    for row in hyperparameter_rows:
        assert row["sigma_f"] > 0.0
        assert row["ell_t"] > 0.0
        assert row["ell_x"] > 0.0
        assert np.isfinite(row["sigma_f"])
        assert np.isfinite(row["ell_t"])
        assert np.isfinite(row["ell_x"])


def test_write_outputs_creates_expected_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    rows = run_experiment(config)

    write_outputs(config, *rows)

    expected_files = {
        "aggregate_metrics.csv",
        "maturity_metrics.csv",
        "predictions.csv",
        "hyperparameters.csv",
        "config.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with (tmp_path / "config.json").open(encoding="utf-8") as handle:
        config_data = json.load(handle)

    assert len(prediction_rows) == 3
    assert "theta_reference" in prediction_rows[0]
    assert config_data["grid_sampling"] == "log_price_uniform"
    assert "gp.predict_greeks" in config_data["greek_convention_note"]
    assert "No finite differences" in config_data["greek_convention_note"]
