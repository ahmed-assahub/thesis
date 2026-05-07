"""Merton GPR grid-size and Monte Carlo path-count comparison experiment."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from option_gpr.benchmarks import (  # noqa: E402
    merton_jump_call_price_log,
    merton_jump_call_price_mc_log,
)
from option_gpr.grids import (  # noqa: E402
    GridSet,
    combine_boundary_points,
    make_random_log_price_interior_points,
    make_random_log_price_lower_boundary,
    make_random_log_price_terminal_boundary,
    make_random_log_price_upper_boundary,
    make_random_price_interior_points,
    make_random_price_lower_boundary,
    make_random_price_terminal_boundary,
    make_random_price_upper_boundary,
)
from option_gpr.hyperparams import ResidualTuningResult, tune_rbf_kernel_residual  # noqa: E402
from option_gpr.kernels import RBFKernel  # noqa: E402
from option_gpr.metrics import mae, max_abs_error, mean_relative_error  # noqa: E402
from option_gpr.models import MertonJumpDiffusionModel  # noqa: E402
from option_gpr.operators import MJDOperator  # noqa: E402
from option_gpr.payoffs import call_boundary_values_log  # noqa: E402
from option_gpr.posterior import StackedOperatorGP  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Merton grid/Monte Carlo comparison experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    jump_intensity: float = 1.0
    jump_mean: float = -0.1
    jump_std: float = 0.2
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    S_min: float = 1.0
    S_max: float = 350.0
    validation_S_min: float = 50.0
    validation_S_max: float = 200.0
    validation_S_step: float = 2
    m_values: tuple[int, ...] = (400, 600, 800)
    mc_path_values: tuple[int, ...] = (500,)
    initial_sigma_f: float = 100.0
    fixed_sigma_f: float | None = 1.0
    initial_ell_t: float = 0.8
    initial_ell_x: float = 1.2
    noise_int: float = 1e-2
    noise_bd: float = 1e-2
    jitter: float = 0.0
    maxiter: int = 300
    xatol: float = 1e-3
    fatol: float = 1e-3
    tail_tol: float = 1e-12
    max_terms: int = 600
    base_seed: int = 20260605
    output_dir: Path = Path("results/merton_jump_diffusion/grid_mc_comparisonlog")
    grid_sampling: str = "log_price_uniform"
    schema_version: str = "1"


@dataclass(frozen=True)
class NestedGridPool:
    """Largest nested random grid for one maturity."""

    X_int: NDArray[np.float64]
    X_terminal: NDArray[np.float64]
    X_lower: NDArray[np.float64]
    X_upper: NDArray[np.float64]


@dataclass(frozen=True)
class SeriesReference:
    """Merton series reference prices for one maturity."""

    maturity: float
    S0: NDArray[np.float64]
    x0: NDArray[np.float64]
    X_star: NDArray[np.float64]
    reference: NDArray[np.float64]
    benchmark_time_sec: float
    rows: list[dict[str, Any]]


def split_boundary_count(n_bd: int) -> tuple[int, int, int]:
    """Split total boundary points into terminal, lower, and upper counts."""

    if n_bd < 3:
        raise ValueError(f"n_bd must be at least 3, got {n_bd!r}.")
    q, rem = divmod(n_bd, 3)
    return q + rem, q, q


def make_boundary_value_fn(config: ExperimentConfig, maturity: float):
    """Return call boundary values for the Merton finite domain."""

    return lambda X: call_boundary_values_log(
        X,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        S_min=config.S_min,
        S_max=config.S_max,
    )


def make_nested_grid_pool(
    config: ExperimentConfig,
    maturity: float,
    max_m: int,
    seed: int,
) -> NestedGridPool:
    """Generate largest nested train or tune pool for one maturity."""

    n_terminal, n_lower, n_upper = _max_boundary_counts(config)
    generators = _grid_point_generators(config.grid_sampling)
    rng = np.random.default_rng(seed)
    return NestedGridPool(
        X_int=generators["interior"](
            max_m,
            t_min=0.0,
            t_max=maturity,
            S_min=config.S_min,
            S_max=config.S_max,
            maturity=maturity,
            rng=rng,
        ),
        X_terminal=generators["terminal"](
            n_terminal,
            maturity=maturity,
            S_min=config.S_min,
            S_max=config.S_max,
            rng=rng,
        ),
        X_lower=generators["lower"](
            n_lower,
            t_min=0.0,
            maturity=maturity,
            S_min=config.S_min,
            rng=rng,
        ),
        X_upper=generators["upper"](
            n_upper,
            t_min=0.0,
            maturity=maturity,
            S_max=config.S_max,
            rng=rng,
        ),
    )


def trim_nested_grid(
    config: ExperimentConfig,
    maturity: float,
    pool: NestedGridPool,
    m: int,
) -> GridSet:
    """Trim a largest nested pool down to the requested ``m=d`` grid."""

    n_terminal, n_lower, n_upper = split_boundary_count(m)
    X_terminal = pool.X_terminal[:n_terminal]
    X_lower = pool.X_lower[:n_lower]
    X_upper = pool.X_upper[:n_upper]
    X_bd = combine_boundary_points(X_terminal, X_lower, X_upper)
    y_bd = make_boundary_value_fn(config, maturity)(X_bd)
    return GridSet(X_int=pool.X_int[:m], X_bd=X_bd, y_bd=y_bd)


def series_reference_for_maturity(
    config: ExperimentConfig,
    maturity: float,
) -> SeriesReference:
    """Return validation grid and Merton series reference prices."""

    S0 = validation_spots(config)
    x0 = np.log(S0)
    X_star = np.column_stack([np.zeros_like(S0), x0])
    start = perf_counter()
    reference = merton_jump_call_price_log(
        X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
        jump_intensity=config.jump_intensity,
        jump_mean=config.jump_mean,
        jump_std=config.jump_std,
        tail_tol=config.tail_tol,
        max_terms=config.max_terms,
    )
    benchmark_time = perf_counter() - start
    rows = [
        {
            "maturity": maturity,
            "S0": float(S),
            "x0": float(x),
            "reference_series": float(ref),
            "series_benchmark_time_sec": benchmark_time,
        }
        for S, x, ref in zip(S0, x0, reference, strict=True)
    ]
    return SeriesReference(
        maturity=maturity,
        S0=S0,
        x0=x0,
        X_star=X_star,
        reference=reference,
        benchmark_time_sec=benchmark_time,
        rows=rows,
    )


def run_single_gpr_case(
    config: ExperimentConfig,
    maturity: float,
    m: int,
    train_grid: GridSet,
    tune_grid: GridSet,
    reference_bundle: SeriesReference,
    train_seed: int,
    tune_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, NDArray[np.float64]],
]:
    """Run one GPR maturity/grid-size case."""

    n_int = m
    n_bd = m
    n_terminal, n_lower, n_upper = split_boundary_count(n_bd)
    model = _make_model(config, maturity)

    opt_start = perf_counter()
    tuning = tune_rbf_kernel_residual(
        model=model,
        operator_factory=_make_mjd_operator,
        train_grid=train_grid,
        tune_grid=tune_grid,
        initial_sigma_f=config.initial_sigma_f,
        initial_ell_t=config.initial_ell_t,
        initial_ell_x=config.initial_ell_x,
        noise_int=config.noise_int,
        noise_bd=config.noise_bd,
        jitter=config.jitter,
        maxiter=config.maxiter,
        xatol=config.xatol,
        fatol=config.fatol,
        fixed_sigma_f=config.fixed_sigma_f,
    )
    optimization_time = perf_counter() - opt_start

    fit_start = perf_counter()
    kernel = RBFKernel(ell_t=tuning.ell_t, ell_x=tuning.ell_x, sigma_f=tuning.sigma_f)
    operator = MJDOperator(model=model, kernel=kernel)
    gp = StackedOperatorGP(
        model=model,
        kernel=kernel,
        operator=operator,
        noise_int=config.noise_int,
        noise_bd=config.noise_bd,
        jitter=config.jitter,
    )
    gp.fit(train_grid.X_int, train_grid.X_bd, train_grid.y_bd)
    final_fit_time = perf_counter() - fit_start

    pricing_start = perf_counter()
    pred = gp.predict(reference_bundle.X_star)
    pricing_time = perf_counter() - pricing_start

    std_start = perf_counter()
    _, var = gp.predict(reference_bundle.X_star, return_var=True)
    posterior_std = np.sqrt(np.maximum(var, 0.0))
    posterior_std_time = perf_counter() - std_start

    metric_values = compute_metrics(
        reference_bundle.S0,
        pred,
        reference_bundle.reference,
        config.strike,
    )
    total_gpr_time = (
        optimization_time + final_fit_time + pricing_time + posterior_std_time
    )

    maturity_row = {
        "m": m,
        "n_int": n_int,
        "n_bd": n_bd,
        "maturity": maturity,
        "n_terminal": n_terminal,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "train_seed": train_seed,
        "tune_seed": tune_seed,
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_posterior_std": float(np.mean(posterior_std)),
        "max_posterior_std": float(np.max(posterior_std)),
        "optimization_time_sec": optimization_time,
        "final_fit_time_sec": final_fit_time,
        "pricing_time_sec": pricing_time,
        "posterior_std_time_sec": posterior_std_time,
        "total_gpr_time_sec": total_gpr_time,
        "series_benchmark_time_sec": reference_bundle.benchmark_time_sec,
        "tuning_success": tuning.success,
        "tuning_message": tuning.message,
        "tuning_nit": tuning.nit,
        "tuning_nfev": tuning.nfev,
        "tuning_objective_value": tuning.objective_value,
    }
    hyperparameter_row = _gpr_hyperparameter_row(maturity, m, n_int, n_bd, tuning, config)
    prediction_rows = _gpr_prediction_rows(
        maturity=maturity,
        m=m,
        n_int=n_int,
        n_bd=n_bd,
        S0=reference_bundle.S0,
        x0=reference_bundle.x0,
        pred=pred,
        ref=reference_bundle.reference,
        posterior_std=posterior_std,
        strike=config.strike,
    )
    arrays = {
        "S0": reference_bundle.S0,
        "pred": pred,
        "ref": reference_bundle.reference,
        "posterior_std": posterior_std,
    }
    return maturity_row, hyperparameter_row, prediction_rows, arrays


def run_single_mc_case(
    config: ExperimentConfig,
    maturity: float,
    n_paths: int,
    reference_bundle: SeriesReference,
    mc_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray[np.float64]]]:
    """Run one Monte Carlo maturity/path-count case."""

    start = perf_counter()
    pred, standard_errors = merton_jump_call_price_mc_log(
        reference_bundle.X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
        jump_intensity=config.jump_intensity,
        jump_mean=config.jump_mean,
        jump_std=config.jump_std,
        n_paths=n_paths,
        seed=mc_seed,
        return_std_error=True,
    )
    mc_time = perf_counter() - start
    metric_values = compute_metrics(
        reference_bundle.S0,
        pred,
        reference_bundle.reference,
        config.strike,
    )
    maturity_row = {
        "n_paths": n_paths,
        "maturity": maturity,
        "mc_seed": mc_seed,
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_mc_standard_error": float(np.mean(standard_errors)),
        "max_mc_standard_error": float(np.max(standard_errors)),
        "mc_time_sec": mc_time,
        "series_benchmark_time_sec": reference_bundle.benchmark_time_sec,
    }
    prediction_rows = _mc_prediction_rows(
        n_paths=n_paths,
        maturity=maturity,
        S0=reference_bundle.S0,
        x0=reference_bundle.x0,
        pred=pred,
        ref=reference_bundle.reference,
        standard_errors=standard_errors,
        strike=config.strike,
    )
    arrays = {
        "S0": reference_bundle.S0,
        "pred": pred,
        "ref": reference_bundle.reference,
        "standard_errors": standard_errors,
    }
    return maturity_row, prediction_rows, arrays


def run_experiment(
    config: ExperimentConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run all configured Merton GPR and Monte Carlo comparisons."""

    gpr_aggregate_inputs: dict[int, dict[str, Any]] = {}
    mc_aggregate_inputs: dict[int, dict[str, Any]] = {}
    gpr_maturity_rows: list[dict[str, Any]] = []
    gpr_hyperparameter_rows: list[dict[str, Any]] = []
    gpr_prediction_rows: list[dict[str, Any]] = []
    mc_maturity_rows: list[dict[str, Any]] = []
    mc_prediction_rows: list[dict[str, Any]] = []
    series_reference_rows: list[dict[str, Any]] = []
    max_m = max(config.m_values)

    for maturity_index, maturity in enumerate(config.maturities):
        reference_bundle = series_reference_for_maturity(config, maturity)
        series_reference_rows.extend(reference_bundle.rows)

        train_seed = config.base_seed + 100 * maturity_index + 1
        tune_seed = config.base_seed + 100 * maturity_index + 2
        train_pool = make_nested_grid_pool(config, maturity, max_m, train_seed)
        tune_pool = make_nested_grid_pool(config, maturity, max_m, tune_seed)
        for m in config.m_values:
            train_grid = trim_nested_grid(config, maturity, train_pool, m)
            tune_grid = trim_nested_grid(config, maturity, tune_pool, m)
            maturity_row, hyper_row, pred_rows, arrays = run_single_gpr_case(
                config,
                maturity=maturity,
                m=m,
                train_grid=train_grid,
                tune_grid=tune_grid,
                reference_bundle=reference_bundle,
                train_seed=train_seed,
                tune_seed=tune_seed,
            )
            gpr_maturity_rows.append(maturity_row)
            gpr_hyperparameter_rows.append(hyper_row)
            gpr_prediction_rows.extend(pred_rows)
            bucket = gpr_aggregate_inputs.setdefault(
                m,
                {
                    "rows": [],
                    "predictions": [],
                    "references": [],
                    "spots": [],
                    "posterior_stds": [],
                },
            )
            bucket["rows"].append(maturity_row)
            bucket["predictions"].append(arrays["pred"])
            bucket["references"].append(arrays["ref"])
            bucket["spots"].append(arrays["S0"])
            bucket["posterior_stds"].append(arrays["posterior_std"])

        for path_index, n_paths in enumerate(config.mc_path_values):
            mc_seed = config.base_seed + 10_000 + 100 * path_index + maturity_index
            maturity_row, pred_rows, arrays = run_single_mc_case(
                config,
                maturity=maturity,
                n_paths=n_paths,
                reference_bundle=reference_bundle,
                mc_seed=mc_seed,
            )
            mc_maturity_rows.append(maturity_row)
            mc_prediction_rows.extend(pred_rows)
            bucket = mc_aggregate_inputs.setdefault(
                n_paths,
                {
                    "rows": [],
                    "predictions": [],
                    "references": [],
                    "spots": [],
                    "standard_errors": [],
                },
            )
            bucket["rows"].append(maturity_row)
            bucket["predictions"].append(arrays["pred"])
            bucket["references"].append(arrays["ref"])
            bucket["spots"].append(arrays["S0"])
            bucket["standard_errors"].append(arrays["standard_errors"])

    gpr_aggregate_rows = [
        _gpr_aggregate_row(
            config=config,
            m=m,
            maturity_rows=bucket["rows"],
            predictions=np.concatenate(bucket["predictions"]),
            references=np.concatenate(bucket["references"]),
            spots=np.concatenate(bucket["spots"]),
            posterior_stds=np.concatenate(bucket["posterior_stds"]),
        )
        for m, bucket in gpr_aggregate_inputs.items()
    ]
    mc_aggregate_rows = [
        _mc_aggregate_row(
            config=config,
            n_paths=n_paths,
            maturity_rows=bucket["rows"],
            predictions=np.concatenate(bucket["predictions"]),
            references=np.concatenate(bucket["references"]),
            spots=np.concatenate(bucket["spots"]),
            standard_errors=np.concatenate(bucket["standard_errors"]),
        )
        for n_paths, bucket in mc_aggregate_inputs.items()
    ]
    return (
        gpr_aggregate_rows,
        gpr_maturity_rows,
        gpr_hyperparameter_rows,
        gpr_prediction_rows,
        mc_aggregate_rows,
        mc_maturity_rows,
        mc_prediction_rows,
        series_reference_rows,
    )


def write_outputs(
    config: ExperimentConfig,
    gpr_aggregate_rows: list[dict[str, Any]],
    gpr_maturity_rows: list[dict[str, Any]],
    gpr_hyperparameter_rows: list[dict[str, Any]],
    gpr_prediction_rows: list[dict[str, Any]],
    mc_aggregate_rows: list[dict[str, Any]],
    mc_maturity_rows: list[dict[str, Any]],
    mc_prediction_rows: list[dict[str, Any]],
    series_reference_rows: list[dict[str, Any]],
) -> None:
    """Write experiment output CSV and JSON files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "gpr_aggregate_metrics.csv", gpr_aggregate_rows)
    _write_csv(config.output_dir / "gpr_maturity_metrics.csv", gpr_maturity_rows)
    _write_csv(config.output_dir / "gpr_hyperparameters.csv", gpr_hyperparameter_rows)
    _write_csv(config.output_dir / "gpr_predictions.csv", gpr_prediction_rows)
    _write_csv(config.output_dir / "mc_aggregate_metrics.csv", mc_aggregate_rows)
    _write_csv(config.output_dir / "mc_maturity_metrics.csv", mc_maturity_rows)
    _write_csv(config.output_dir / "mc_predictions.csv", mc_prediction_rows)
    _write_csv(config.output_dir / "series_reference.csv", series_reference_rows)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_json(config), handle, indent=2)


def main() -> None:
    """Run the full experiment and write outputs."""

    config = ExperimentConfig()
    rows = run_experiment(config)
    write_outputs(config, *rows)
    print(f"Wrote results to {config.output_dir}")


def validation_spots(config: ExperimentConfig) -> NDArray[np.float64]:
    """Return validation spots in ordinary price coordinates."""

    return np.arange(
        config.validation_S_min,
        config.validation_S_max + 1e-12,
        config.validation_S_step,
        dtype=float,
    )


def compute_metrics(
    S0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    strike: float,
) -> dict[str, float]:
    """Return MAE, MaxAE, and ITM mean relative error."""

    itm = S0 > strike
    if not np.any(itm):
        raise ValueError("At least one validation spot must be in the money.")
    return {
        "mae": mae(pred, ref),
        "max_abs_error": max_abs_error(pred, ref),
        "mre_itm": mean_relative_error(pred[itm], ref[itm]),
    }


def _make_model(config: ExperimentConfig, maturity: float) -> MertonJumpDiffusionModel:
    return MertonJumpDiffusionModel(
        r=config.r,
        sigma=config.sigma,
        jump_intensity=config.jump_intensity,
        jump_mean=config.jump_mean,
        jump_std=config.jump_std,
        strike=config.strike,
        maturity=maturity,
    )


def _make_mjd_operator(
    model: MertonJumpDiffusionModel, kernel: RBFKernel
) -> MJDOperator:
    return MJDOperator(model=model, kernel=kernel)


def _max_boundary_counts(config: ExperimentConfig) -> tuple[int, int, int]:
    counts = [split_boundary_count(m) for m in config.m_values]
    return tuple(max(parts) for parts in zip(*counts, strict=True))


def _grid_point_generators(grid_sampling: str) -> dict[str, Any]:
    if grid_sampling == "price_uniform":
        return {
            "interior": make_random_price_interior_points,
            "terminal": make_random_price_terminal_boundary,
            "lower": make_random_price_lower_boundary,
            "upper": make_random_price_upper_boundary,
        }
    if grid_sampling == "log_price_uniform":
        return {
            "interior": make_random_log_price_interior_points,
            "terminal": make_random_log_price_terminal_boundary,
            "lower": make_random_log_price_lower_boundary,
            "upper": make_random_log_price_upper_boundary,
        }
    raise ValueError(
        "grid_sampling must be 'price_uniform' or 'log_price_uniform', "
        f"got {grid_sampling!r}."
    )


def _gpr_aggregate_row(
    *,
    config: ExperimentConfig,
    m: int,
    maturity_rows: list[dict[str, Any]],
    predictions: NDArray[np.float64],
    references: NDArray[np.float64],
    spots: NDArray[np.float64],
    posterior_stds: NDArray[np.float64],
) -> dict[str, Any]:
    n_terminal, n_lower, n_upper = split_boundary_count(m)
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    objectives = np.array([row["tuning_objective_value"] for row in maturity_rows])
    successes = [bool(row["tuning_success"]) for row in maturity_rows]
    return {
        "m": m,
        "n_int": m,
        "n_bd": m,
        "n_terminal": n_terminal,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "n_maturities": len(config.maturities),
        "n_validation_spots": validation_spots(config).shape[0],
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_posterior_std": float(np.mean(posterior_stds)),
        "max_posterior_std": float(np.max(posterior_stds)),
        "optimization_time_sec": sum(row["optimization_time_sec"] for row in maturity_rows),
        "final_fit_time_sec": sum(row["final_fit_time_sec"] for row in maturity_rows),
        "pricing_time_sec": sum(row["pricing_time_sec"] for row in maturity_rows),
        "posterior_std_time_sec": sum(
            row["posterior_std_time_sec"] for row in maturity_rows
        ),
        "total_gpr_time_sec": sum(row["total_gpr_time_sec"] for row in maturity_rows),
        "series_benchmark_time_sec": sum(
            row["series_benchmark_time_sec"] for row in maturity_rows
        ),
        "tuning_success_count": sum(successes),
        "tuning_failure_count": len(successes) - sum(successes),
        "mean_objective_value": float(np.mean(objectives)),
        "max_objective_value": float(np.max(objectives)),
    }


def _mc_aggregate_row(
    *,
    config: ExperimentConfig,
    n_paths: int,
    maturity_rows: list[dict[str, Any]],
    predictions: NDArray[np.float64],
    references: NDArray[np.float64],
    spots: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
) -> dict[str, Any]:
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    return {
        "n_paths": n_paths,
        "n_maturities": len(config.maturities),
        "n_validation_spots": validation_spots(config).shape[0],
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_mc_standard_error": float(np.mean(standard_errors)),
        "max_mc_standard_error": float(np.max(standard_errors)),
        "mc_time_sec": sum(row["mc_time_sec"] for row in maturity_rows),
        "series_benchmark_time_sec": sum(
            row["series_benchmark_time_sec"] for row in maturity_rows
        ),
    }


def _gpr_hyperparameter_row(
    maturity: float,
    m: int,
    n_int: int,
    n_bd: int,
    tuning: ResidualTuningResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "m": m,
        "n_int": n_int,
        "n_bd": n_bd,
        "maturity": maturity,
        "sigma_f": tuning.sigma_f,
        "ell_t": tuning.ell_t,
        "ell_x": tuning.ell_x,
        "log_sigma_f": "" if config.fixed_sigma_f is not None else tuning.theta_log[0],
        "log_ell_t": tuning.theta_log[0] if config.fixed_sigma_f is not None else tuning.theta_log[1],
        "log_ell_x": tuning.theta_log[1] if config.fixed_sigma_f is not None else tuning.theta_log[2],
        "initial_sigma_f": config.initial_sigma_f,
        "fixed_sigma_f": "" if config.fixed_sigma_f is None else config.fixed_sigma_f,
        "initial_ell_t": config.initial_ell_t,
        "initial_ell_x": config.initial_ell_x,
        "objective_value": tuning.objective_value,
        "success": tuning.success,
        "nit": tuning.nit,
        "nfev": tuning.nfev,
    }


def _gpr_prediction_rows(
    *,
    maturity: float,
    m: int,
    n_int: int,
    n_bd: int,
    S0: NDArray[np.float64],
    x0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    posterior_std: NDArray[np.float64],
    strike: float,
) -> list[dict[str, Any]]:
    return [
        {
            "method": "gpr",
            "m": m,
            "n_int": n_int,
            "n_bd": n_bd,
            "maturity": maturity,
            "S0": float(S),
            "x0": float(x),
            "prediction": float(pred_i),
            "reference_series": float(ref_i),
            "spread": float(pred_i - ref_i),
            "abs_error": float(abs(pred_i - ref_i)),
            "is_itm": bool(S > strike),
            "posterior_std": float(std_i),
        }
        for S, x, pred_i, ref_i, std_i in zip(
            S0, x0, pred, ref, posterior_std, strict=True
        )
    ]


def _mc_prediction_rows(
    *,
    n_paths: int,
    maturity: float,
    S0: NDArray[np.float64],
    x0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    strike: float,
) -> list[dict[str, Any]]:
    return [
        {
            "method": "mc",
            "n_paths": n_paths,
            "maturity": maturity,
            "S0": float(S),
            "x0": float(x),
            "prediction": float(pred_i),
            "reference_series": float(ref_i),
            "spread": float(pred_i - ref_i),
            "abs_error": float(abs(pred_i - ref_i)),
            "is_itm": bool(S > strike),
            "mc_standard_error": float(se_i),
        }
        for S, x, pred_i, ref_i, se_i in zip(
            S0, x0, pred, ref, standard_errors, strict=True
        )
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _config_json(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["created_at_utc"] = datetime.now(UTC).isoformat()
    data["grid_design"] = "nested_random_price_fixed_ratio"
    data["grid_design_note"] = (
        "For each maturity, largest train/tune pools are generated once. "
        "Requested GPR grid sizes are prefixes of the same interior, terminal, "
        "lower, and upper blocks."
    )
    data["grid_sampling_note"] = (
        "'price_uniform' samples uniformly in S and returns x = log(S). "
        "'log_price_uniform' samples uniformly directly in x = log(S)."
    )
    data["gpr_seed_policy"] = (
        "train_seed = base_seed + 100 * maturity_index + 1; "
        "tune_seed = base_seed + 100 * maturity_index + 2"
    )
    data["mc_seed_policy"] = (
        "mc_seed = base_seed + 10000 + 100 * path_index + maturity_index"
    )
    data["error_reference_note"] = (
        "All GPR and Monte Carlo errors are computed against the Merton "
        "semi-analytical series benchmark, never against each other."
    )
    data["runtime_definitions"] = {
        "optimization_time_sec": "residual hyperparameter tuning only",
        "final_fit_time_sec": "final GP fit after tuning",
        "pricing_time_sec": "posterior mean prediction on validation set",
        "posterior_std_time_sec": "posterior variance/std prediction on validation set",
        "total_gpr_time_sec": (
            "optimization_time_sec + final_fit_time_sec + pricing_time_sec + "
            "posterior_std_time_sec"
        ),
        "mc_time_sec": "Monte Carlo prices and standard errors on validation set",
        "series_benchmark_time_sec": (
            "Merton series reference evaluation time, stored separately"
        ),
    }
    return data


if __name__ == "__main__":
    main()
