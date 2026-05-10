"""Merton single-surface GPR and Monte Carlo comparison experiment."""

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
    make_random_log_price_grid,
    make_random_price_grid,
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
    """Configuration for the Merton single-surface comparison experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    jump_intensity: float = 1.0
    jump_mean: float = -0.1
    jump_std: float = 0.2
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    S_min: float = 1.0
    S_max: float = 350.0
    validation_S_min: float = 50.0
    validation_S_max: float = 200.0
    validation_S_step: float = 2.0
    n_int: int = 1000
    n_bd: int = 1000
    mc_path_values: tuple[int, ...] = (5000, )
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
    max_terms: int = 200
    base_seed: int = 20290701
    output_dir: Path = Path("results/merton_jump_diffusion/single_surface_mc_comparison")
    grid_sampling: str = "price_uniform"
    schema_version: str = "1"


@dataclass(frozen=True)
class SeriesReference:
    """Merton series reference prices for one remaining maturity."""

    remaining_maturity: float
    S0: NDArray[np.float64]
    x0: NDArray[np.float64]
    X_ref: NDArray[np.float64]
    reference: NDArray[np.float64]
    benchmark_time_sec: float
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class TrainedSurface:
    """One fitted GPR posterior on the largest maturity interval."""

    gp: StackedOperatorGP
    tuning: ResidualTuningResult
    T_max: float
    train_seed: int
    tune_seed: int
    optimization_time_sec: float
    final_fit_time_sec: float


def split_boundary_count(n_bd: int) -> tuple[int, int, int]:
    """Split total boundary points into terminal, lower, and upper counts."""

    if n_bd < 3:
        raise ValueError(f"n_bd must be at least 3, got {n_bd!r}.")
    q, rem = divmod(n_bd, 3)
    return q + rem, q, q


def make_grid(config: ExperimentConfig, maturity: float, seed: int) -> GridSet:
    """Generate one random train or tune grid for the largest maturity."""

    n_terminal, n_lower, n_upper = split_boundary_count(config.n_bd)
    grid_factory = _grid_factory(config.grid_sampling)
    return grid_factory(
        n_int=config.n_int,
        n_terminal=n_terminal,
        n_lower=n_lower,
        n_upper=n_upper,
        t_min=0.0,
        t_max=maturity,
        maturity=maturity,
        S_min=config.S_min,
        S_max=config.S_max,
        boundary_value_fn=make_boundary_value_fn(config, maturity),
        seed=seed,
    )


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


def train_single_gpr_surface(config: ExperimentConfig) -> TrainedSurface:
    """Tune and fit one GPR posterior on ``[0, T_max]``."""

    T_max = max(config.maturities)
    train_seed = config.base_seed + 1
    tune_seed = config.base_seed + 2
    train_grid = make_grid(config, T_max, train_seed)
    tune_grid = make_grid(config, T_max, tune_seed)
    model = _make_model(config, T_max)

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

    return TrainedSurface(
        gp=gp,
        tuning=tuning,
        T_max=T_max,
        train_seed=train_seed,
        tune_seed=tune_seed,
        optimization_time_sec=optimization_time,
        final_fit_time_sec=final_fit_time,
    )


def series_reference_for_maturity(
    config: ExperimentConfig,
    remaining_maturity: float,
) -> SeriesReference:
    """Return validation grid and Merton series prices for one maturity."""

    S0 = validation_spots(config)
    x0 = np.log(S0)
    X_ref = np.column_stack([np.zeros_like(S0), x0])
    start = perf_counter()
    reference = merton_jump_call_price_log(
        X_ref,
        strike=config.strike,
        maturity=remaining_maturity,
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
            "remaining_maturity": remaining_maturity,
            "S0": float(S),
            "x0": float(x),
            "reference_series": float(ref),
            "series_benchmark_time_sec": benchmark_time,
        }
        for S, x, ref in zip(S0, x0, reference, strict=True)
    ]
    return SeriesReference(
        remaining_maturity=remaining_maturity,
        S0=S0,
        x0=x0,
        X_ref=X_ref,
        reference=reference,
        benchmark_time_sec=benchmark_time,
        rows=rows,
    )


def run_gpr_evaluation(
    config: ExperimentConfig,
    trained_surface: TrainedSurface,
    reference_bundles: list[SeriesReference],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate one trained GPR surface at all requested remaining maturities."""

    maturity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    all_predictions: list[NDArray[np.float64]] = []
    all_references: list[NDArray[np.float64]] = []
    all_spots: list[NDArray[np.float64]] = []
    all_posterior_stds: list[NDArray[np.float64]] = []
    total_pricing_time = 0.0
    total_std_time = 0.0

    for reference_bundle in reference_bundles:
        eval_t = trained_surface.T_max - reference_bundle.remaining_maturity
        X_gpr = np.column_stack(
            [np.full_like(reference_bundle.S0, eval_t), reference_bundle.x0]
        )

        pricing_start = perf_counter()
        pred = trained_surface.gp.predict(X_gpr)
        pricing_time = perf_counter() - pricing_start
        total_pricing_time += pricing_time

        std_start = perf_counter()
        _, var = trained_surface.gp.predict(X_gpr, return_var=True)
        posterior_std = np.sqrt(np.maximum(var, 0.0))
        posterior_std_time = perf_counter() - std_start
        total_std_time += posterior_std_time

        metric_values = compute_metrics(
            reference_bundle.S0,
            pred,
            reference_bundle.reference,
            config.strike,
        )
        maturity_rows.append(
            _gpr_maturity_row(
                config=config,
                trained_surface=trained_surface,
                remaining_maturity=reference_bundle.remaining_maturity,
                eval_t=eval_t,
                pred=pred,
                ref=reference_bundle.reference,
                S0=reference_bundle.S0,
                posterior_std=posterior_std,
                pricing_time=pricing_time,
                posterior_std_time=posterior_std_time,
                series_time=reference_bundle.benchmark_time_sec,
                metric_values=metric_values,
            )
        )
        prediction_rows.extend(
            _gpr_prediction_rows(
                remaining_maturity=reference_bundle.remaining_maturity,
                eval_t=eval_t,
                S0=reference_bundle.S0,
                x0=reference_bundle.x0,
                pred=pred,
                ref=reference_bundle.reference,
                posterior_std=posterior_std,
                strike=config.strike,
            )
        )
        all_predictions.append(pred)
        all_references.append(reference_bundle.reference)
        all_spots.append(reference_bundle.S0)
        all_posterior_stds.append(posterior_std)

    aggregate_rows = [
        _gpr_aggregate_row(
            config=config,
            trained_surface=trained_surface,
            predictions=np.concatenate(all_predictions),
            references=np.concatenate(all_references),
            spots=np.concatenate(all_spots),
            posterior_stds=np.concatenate(all_posterior_stds),
            pricing_time=total_pricing_time,
            posterior_std_time=total_std_time,
            series_time=sum(bundle.benchmark_time_sec for bundle in reference_bundles),
        )
    ]
    hyperparameter_rows = [_gpr_hyperparameter_row(config, trained_surface)]
    return aggregate_rows, maturity_rows, hyperparameter_rows, prediction_rows


def run_single_mc_case(
    config: ExperimentConfig,
    remaining_maturity: float,
    n_paths: int,
    reference_bundle: SeriesReference,
    mc_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray[np.float64]]]:
    """Run one Monte Carlo remaining-maturity/path-count case."""

    start = perf_counter()
    pred, standard_errors = merton_jump_call_price_mc_log(
        reference_bundle.X_ref,
        strike=config.strike,
        maturity=remaining_maturity,
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
        "remaining_maturity": remaining_maturity,
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
        remaining_maturity=remaining_maturity,
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
    """Run the Merton single-surface GPR and Monte Carlo comparison."""

    trained_surface = train_single_gpr_surface(config)
    reference_bundles = [
        series_reference_for_maturity(config, maturity)
        for maturity in config.maturities
    ]
    series_reference_rows = [
        row for bundle in reference_bundles for row in bundle.rows
    ]
    (
        gpr_aggregate_rows,
        gpr_maturity_rows,
        gpr_hyperparameter_rows,
        gpr_prediction_rows,
    ) = run_gpr_evaluation(config, trained_surface, reference_bundles)

    mc_aggregate_inputs: dict[int, dict[str, Any]] = {}
    mc_maturity_rows: list[dict[str, Any]] = []
    mc_prediction_rows: list[dict[str, Any]] = []
    for path_index, n_paths in enumerate(config.mc_path_values):
        for maturity_index, reference_bundle in enumerate(reference_bundles):
            mc_seed = config.base_seed + 10_000 + 100 * path_index + maturity_index
            maturity_row, pred_rows, arrays = run_single_mc_case(
                config,
                remaining_maturity=reference_bundle.remaining_maturity,
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


def _grid_factory(grid_sampling: str):
    if grid_sampling == "price_uniform":
        return make_random_price_grid
    if grid_sampling == "log_price_uniform":
        return make_random_log_price_grid
    raise ValueError(
        "grid_sampling must be 'price_uniform' or 'log_price_uniform', "
        f"got {grid_sampling!r}."
    )


def _gpr_maturity_row(
    *,
    config: ExperimentConfig,
    trained_surface: TrainedSurface,
    remaining_maturity: float,
    eval_t: float,
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    S0: NDArray[np.float64],
    posterior_std: NDArray[np.float64],
    pricing_time: float,
    posterior_std_time: float,
    series_time: float,
    metric_values: dict[str, float],
) -> dict[str, Any]:
    training_time = (
        trained_surface.optimization_time_sec + trained_surface.final_fit_time_sec
    )
    return {
        "remaining_maturity": remaining_maturity,
        "eval_t": eval_t,
        "n_int": config.n_int,
        "n_bd": config.n_bd,
        "train_seed": trained_surface.train_seed,
        "tune_seed": trained_surface.tune_seed,
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_posterior_std": float(np.mean(posterior_std)),
        "max_posterior_std": float(np.max(posterior_std)),
        "optimization_time_sec": trained_surface.optimization_time_sec,
        "final_fit_time_sec": trained_surface.final_fit_time_sec,
        "gpr_training_time_sec": training_time,
        "gpr_pricing_time_sec": pricing_time,
        "posterior_std_time_sec": posterior_std_time,
        "total_gpr_price_time_sec": training_time + pricing_time,
        "total_gpr_with_std_time_sec": training_time + pricing_time + posterior_std_time,
        "series_benchmark_time_sec": series_time,
        "tuning_success": trained_surface.tuning.success,
        "tuning_message": trained_surface.tuning.message,
        "tuning_nit": trained_surface.tuning.nit,
        "tuning_nfev": trained_surface.tuning.nfev,
        "tuning_objective_value": trained_surface.tuning.objective_value,
    }


def _gpr_aggregate_row(
    *,
    config: ExperimentConfig,
    trained_surface: TrainedSurface,
    predictions: NDArray[np.float64],
    references: NDArray[np.float64],
    spots: NDArray[np.float64],
    posterior_stds: NDArray[np.float64],
    pricing_time: float,
    posterior_std_time: float,
    series_time: float,
) -> dict[str, Any]:
    n_terminal, n_lower, n_upper = split_boundary_count(config.n_bd)
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    training_time = (
        trained_surface.optimization_time_sec + trained_surface.final_fit_time_sec
    )
    return {
        "n_int": config.n_int,
        "n_bd": config.n_bd,
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
        "optimization_time_sec": trained_surface.optimization_time_sec,
        "final_fit_time_sec": trained_surface.final_fit_time_sec,
        "gpr_training_time_sec": training_time,
        "gpr_pricing_time_sec": pricing_time,
        "posterior_std_time_sec": posterior_std_time,
        "total_gpr_price_time_sec": training_time + pricing_time,
        "total_gpr_with_std_time_sec": training_time + pricing_time + posterior_std_time,
        "series_benchmark_time_sec": series_time,
        "tuning_success": trained_surface.tuning.success,
        "tuning_message": trained_surface.tuning.message,
        "tuning_nit": trained_surface.tuning.nit,
        "tuning_nfev": trained_surface.tuning.nfev,
        "tuning_objective_value": trained_surface.tuning.objective_value,
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
    config: ExperimentConfig,
    trained_surface: TrainedSurface,
) -> dict[str, Any]:
    tuning = trained_surface.tuning
    return {
        "T_max": trained_surface.T_max,
        "n_int": config.n_int,
        "n_bd": config.n_bd,
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
    remaining_maturity: float,
    eval_t: float,
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
            "remaining_maturity": remaining_maturity,
            "eval_t": eval_t,
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
    remaining_maturity: float,
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
            "remaining_maturity": remaining_maturity,
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
    T_max = max(config.maturities)
    data["output_dir"] = str(config.output_dir)
    data["T_max"] = T_max
    data["created_at_utc"] = datetime.now(UTC).isoformat()
    data["surface_design"] = "single_gpr_surface_on_T_max"
    data["surface_design_note"] = (
        "One GPR posterior is trained on [0, T_max]. Prices for remaining "
        "maturity T_i are evaluated at eval_t = T_max - T_i."
    )
    data["grid_sampling_note"] = (
        "'price_uniform' samples uniformly in S and returns x = log(S). "
        "'log_price_uniform' samples uniformly directly in x = log(S)."
    )
    data["gpr_seed_policy"] = (
        "train_seed = base_seed + 1; tune_seed = base_seed + 2"
    )
    data["mc_seed_policy"] = (
        "mc_seed = base_seed + 10000 + 100 * path_index + maturity_index"
    )
    data["error_reference_note"] = (
        "All GPR and Monte Carlo errors are computed against the Merton "
        "semi-analytical series benchmark, never against each other."
    )
    data["runtime_definitions"] = {
        "optimization_time_sec": "one-time residual hyperparameter tuning",
        "final_fit_time_sec": "one-time final GP fit after tuning",
        "gpr_training_time_sec": "optimization_time_sec + final_fit_time_sec",
        "gpr_pricing_time_sec": "posterior mean prediction over all maturities/spots",
        "posterior_std_time_sec": "posterior variance/std over all maturities/spots",
        "total_gpr_price_time_sec": (
            "gpr_training_time_sec + gpr_pricing_time_sec"
        ),
        "total_gpr_with_std_time_sec": (
            "gpr_training_time_sec + gpr_pricing_time_sec + posterior_std_time_sec"
        ),
        "mc_time_sec": "Monte Carlo prices and standard errors",
        "series_benchmark_time_sec": (
            "Merton series reference evaluation time, stored separately"
        ),
    }
    return data


if __name__ == "__main__":
    main()
