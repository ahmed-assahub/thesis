"""Black-Scholes random-seed stability experiment."""

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

from option_gpr.benchmarks import black_scholes_call_price_log
from option_gpr.grids import GridSet, make_random_log_price_grid, make_random_price_grid
from option_gpr.hyperparams import ResidualTuningResult, tune_rbf_kernel_residual
from option_gpr.kernels import RBFKernel
from option_gpr.metrics import mae, max_abs_error, mean_relative_error
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator
from option_gpr.payoffs import call_boundary_values_log
from option_gpr.posterior import StackedOperatorGP


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Black-Scholes random-seed stability experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    S_min: float = 1.0
    S_max: float = 200.0
    validation_S_min: float = 50.0
    validation_S_max: float = 150.0
    validation_S_step: float = 2.0
    n_seeds: int = 20
    n_int: int = 800
    n_bd: int = 800
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
    global_base_seed: int = 20260520
    output_dir: Path = Path("results/black_scholes/seed_stability_log")
    grid_sampling: str = "log_price_uniform"
    schema_version: str = "1"
    show_progress: bool = True


def split_boundary_count(n_bd: int) -> tuple[int, int, int]:
    """Split total boundary points into terminal, lower, and upper counts."""

    if n_bd < 3:
        raise ValueError(f"n_bd must be at least 3, got {n_bd!r}.")
    q, rem = divmod(n_bd, 3)
    return q + rem, q, q


def make_boundary_value_fn(config: ExperimentConfig, maturity: float):
    """Return boundary value function for Black-Scholes call boundaries."""

    return lambda X: call_boundary_values_log(
        X,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        S_min=config.S_min,
        S_max=config.S_max,
    )


def make_grid(config: ExperimentConfig, maturity: float, seed: int) -> GridSet:
    """Generate one random training or tuning grid."""

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


def run_single_maturity(
    config: ExperimentConfig,
    seed_index: int,
    seed_base: int,
    maturity: float,
    maturity_index: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, NDArray[np.float64]],
]:
    """Run one seed repetition and maturity."""

    train_seed = seed_base + 100 * maturity_index + 1
    tune_seed = seed_base + 100 * maturity_index + 2
    model = BlackScholesModel(
        r=config.r,
        sigma=config.sigma,
        strike=config.strike,
        maturity=maturity,
    )
    train_grid = make_grid(config, maturity, train_seed)
    tune_grid = make_grid(config, maturity, tune_seed)

    opt_start = perf_counter()
    tuning = tune_rbf_kernel_residual(
        model=model,
        operator_factory=_make_bs_operator,
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

    gpr_start = perf_counter()
    kernel = RBFKernel(ell_t=tuning.ell_t, ell_x=tuning.ell_x, sigma_f=tuning.sigma_f)
    operator = BSLogOperator(model=model, kernel=kernel)
    gp = StackedOperatorGP(
        model=model,
        kernel=kernel,
        operator=operator,
        noise_int=config.noise_int,
        noise_bd=config.noise_bd,
        jitter=config.jitter,
    )
    gp.fit(train_grid.X_int, train_grid.X_bd, train_grid.y_bd)

    S0 = validation_spots(config)
    x0 = np.log(S0)
    X_star = np.column_stack([np.zeros_like(S0), x0])
    pred = gp.predict(X_star)
    _, var = gp.predict(X_star, return_var=True)
    posterior_std = np.sqrt(np.maximum(var, 0.0))
    gpr_time = perf_counter() - gpr_start

    benchmark_start = perf_counter()
    ref = black_scholes_call_price_log(
        X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
    )
    benchmark_time = perf_counter() - benchmark_start

    metric_values = compute_metrics(S0, pred, ref, config.strike)
    total_time = optimization_time + gpr_time + benchmark_time

    maturity_row = {
        "seed_index": seed_index,
        "base_seed": seed_base,
        "maturity": maturity,
        "n_int": config.n_int,
        "n_bd": config.n_bd,
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_posterior_std": float(np.mean(posterior_std)),
        "max_posterior_std": float(np.max(posterior_std)),
        "optimization_time_sec": optimization_time,
        "gpr_time_sec": gpr_time,
        "benchmark_time_sec": benchmark_time,
        "total_time_sec": total_time,
        "tuning_success": tuning.success,
        "tuning_message": tuning.message,
        "tuning_nit": tuning.nit,
        "tuning_nfev": tuning.nfev,
        "tuning_objective_value": tuning.objective_value,
    }
    hyperparameter_row = _hyperparameter_row(
        seed_index, seed_base, maturity, tuning, config
    )
    prediction_rows = _prediction_rows(
        seed_index=seed_index,
        seed_base=seed_base,
        maturity=maturity,
        S0=S0,
        x0=x0,
        pred=pred,
        ref=ref,
        posterior_std=posterior_std,
        strike=config.strike,
    )
    arrays = {"S0": S0, "pred": pred, "ref": ref, "posterior_std": posterior_std}
    return maturity_row, hyperparameter_row, prediction_rows, arrays


def run_single_seed(
    config: ExperimentConfig,
    seed_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all maturities for one seed repetition."""

    seed_base = config.global_base_seed + 10000 * seed_index
    maturity_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions: list[NDArray[np.float64]] = []
    references: list[NDArray[np.float64]] = []
    spots: list[NDArray[np.float64]] = []
    posterior_stds: list[NDArray[np.float64]] = []

    for maturity_index, maturity in enumerate(config.maturities):
        maturity_row, hyper_row, pred_rows, arrays = run_single_maturity(
            config,
            seed_index=seed_index,
            seed_base=seed_base,
            maturity=maturity,
            maturity_index=maturity_index,
        )
        maturity_rows.append(maturity_row)
        hyperparameter_rows.append(hyper_row)
        prediction_rows.extend(pred_rows)
        predictions.append(arrays["pred"])
        references.append(arrays["ref"])
        spots.append(arrays["S0"])
        posterior_stds.append(arrays["posterior_std"])

    seed_row = _seed_row(
        config=config,
        seed_index=seed_index,
        seed_base=seed_base,
        maturity_rows=maturity_rows,
        predictions=np.concatenate(predictions),
        references=np.concatenate(references),
        spots=np.concatenate(spots),
        posterior_stds=np.concatenate(posterior_stds),
    )
    return seed_row, maturity_rows, hyperparameter_rows, prediction_rows


def run_experiment(
    config: ExperimentConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run all configured seed repetitions."""

    seed_rows: list[dict[str, Any]] = []
    maturity_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_cases = config.n_seeds * len(config.maturities)
    completed_cases = 0
    experiment_start = perf_counter()
    if config.show_progress:
        print(
            f"Starting Black-Scholes seed-stability experiment: "
            f"{config.n_seeds} seeds x {len(config.maturities)} maturities "
            f"= {total_cases} cases.",
            flush=True,
        )
    for seed_index in range(config.n_seeds):
        seed_start = perf_counter()
        if config.show_progress:
            elapsed = seed_start - experiment_start
            print(
                f"[seed {seed_index + 1}/{config.n_seeds}] "
                f"Starting seed_index={seed_index} | "
                f"elapsed={_format_duration(elapsed)}",
                flush=True,
            )
        seed_row, seed_maturity_rows, seed_hyper_rows, seed_prediction_rows = (
            run_single_seed(config, seed_index)
        )
        seed_rows.append(seed_row)
        maturity_rows.extend(seed_maturity_rows)
        hyperparameter_rows.extend(seed_hyper_rows)
        prediction_rows.extend(seed_prediction_rows)
        completed_cases += len(config.maturities)
        if config.show_progress:
            now = perf_counter()
            elapsed = now - experiment_start
            seed_time = now - seed_start
            avg_case_time = elapsed / completed_cases
            eta = avg_case_time * (total_cases - completed_cases)
            print(
                f"[seed {seed_index + 1}/{config.n_seeds}] "
                f"Done seed_index={seed_index} | "
                f"seed={_format_duration(seed_time)} | "
                f"elapsed={_format_duration(elapsed)} | "
                f"eta={_format_duration(eta)} | "
                f"mae={seed_row['mae']:.6g} | "
                f"failures={seed_row['tuning_failure_count']}",
                flush=True,
            )

    summary_rows = _summary_rows(seed_rows)
    if config.show_progress:
        total_elapsed = perf_counter() - experiment_start
        print(
            f"Completed Black-Scholes seed-stability experiment in "
            f"{_format_duration(total_elapsed)}.",
            flush=True,
        )
    return seed_rows, summary_rows, hyperparameter_rows, maturity_rows, prediction_rows


def write_outputs(
    config: ExperimentConfig,
    seed_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
    maturity_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    """Write experiment output CSV and JSON files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "seed_metrics.csv", seed_rows)
    _write_csv(config.output_dir / "summary_metrics.csv", summary_rows)
    _write_csv(config.output_dir / "hyperparameters.csv", hyperparameter_rows)
    _write_csv(config.output_dir / "maturity_metrics.csv", maturity_rows)
    _write_csv(config.output_dir / "predictions.csv", prediction_rows)
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


def _make_bs_operator(model: BlackScholesModel, kernel: RBFKernel) -> BSLogOperator:
    return BSLogOperator(model=model, kernel=kernel)


def _grid_factory(grid_sampling: str):
    if grid_sampling == "price_uniform":
        return make_random_price_grid
    if grid_sampling == "log_price_uniform":
        return make_random_log_price_grid
    raise ValueError(
        "grid_sampling must be 'price_uniform' or 'log_price_uniform', "
        f"got {grid_sampling!r}."
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _seed_row(
    *,
    config: ExperimentConfig,
    seed_index: int,
    seed_base: int,
    maturity_rows: list[dict[str, Any]],
    predictions: NDArray[np.float64],
    references: NDArray[np.float64],
    spots: NDArray[np.float64],
    posterior_stds: NDArray[np.float64],
) -> dict[str, Any]:
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    objectives = np.array([row["tuning_objective_value"] for row in maturity_rows])
    successes = [bool(row["tuning_success"]) for row in maturity_rows]
    return {
        "seed_index": seed_index,
        "base_seed": seed_base,
        "n_int": config.n_int,
        "n_bd": config.n_bd,
        "n_maturities": len(config.maturities),
        "n_validation_spots": validation_spots(config).shape[0],
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "mean_posterior_std": float(np.mean(posterior_stds)),
        "max_posterior_std": float(np.max(posterior_stds)),
        "optimization_time_sec": sum(row["optimization_time_sec"] for row in maturity_rows),
        "gpr_time_sec": sum(row["gpr_time_sec"] for row in maturity_rows),
        "benchmark_time_sec": sum(row["benchmark_time_sec"] for row in maturity_rows),
        "total_time_sec": sum(row["total_time_sec"] for row in maturity_rows),
        "tuning_success_count": sum(successes),
        "tuning_failure_count": len(successes) - sum(successes),
        "mean_objective_value": float(np.mean(objectives)),
        "max_objective_value": float(np.max(objectives)),
    }


def _summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "mae",
        "max_abs_error",
        "mre_itm",
        "mean_posterior_std",
        "max_posterior_std",
        "optimization_time_sec",
        "gpr_time_sec",
        "benchmark_time_sec",
        "total_time_sec",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = np.array([row[metric] for row in seed_rows], dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return rows


def _hyperparameter_row(
    seed_index: int,
    seed_base: int,
    maturity: float,
    tuning: ResidualTuningResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "seed_index": seed_index,
        "base_seed": seed_base,
        "maturity": maturity,
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


def _prediction_rows(
    *,
    seed_index: int,
    seed_base: int,
    maturity: float,
    S0: NDArray[np.float64],
    x0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    posterior_std: NDArray[np.float64],
    strike: float,
) -> list[dict[str, Any]]:
    return [
        {
            "seed_index": seed_index,
            "base_seed": seed_base,
            "maturity": maturity,
            "S0": float(S),
            "x0": float(x),
            "prediction": float(pred_i),
            "reference": float(ref_i),
            "abs_error": float(abs(pred_i - ref_i)),
            "is_itm": bool(S > strike),
            "posterior_std": float(std_i),
        }
        for S, x, pred_i, ref_i, std_i in zip(
            S0, x0, pred, ref, posterior_std, strict=True
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
    data["seed_policy"] = (
        "seed_base = global_base_seed + 10000 * seed_index; "
        "train_seed = seed_base + 100 * maturity_index + 1; "
        "tune_seed = seed_base + 100 * maturity_index + 2"
    )
    data["grid_sampling_note"] = (
        "'price_uniform' samples uniformly in S and returns x = log(S). "
        "'log_price_uniform' samples uniformly directly in x = log(S)."
    )
    data["execution"] = (
        "Sequential by default. Parallel execution is intentionally omitted to "
        "avoid BLAS thread oversubscription and thermal throttling on fanless "
        "laptops; measure sequential runtime before adding optional n_jobs."
    )
    data["prediction_rows"] = config.n_seeds * len(config.maturities) * validation_spots(config).shape[0]
    return data


if __name__ == "__main__":
    main()
