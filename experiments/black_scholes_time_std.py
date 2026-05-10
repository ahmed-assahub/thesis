"""Black-Scholes fixed-ratio timing and posterior-std experiment."""

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
from option_gpr.grids import (
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
from option_gpr.hyperparams import ResidualTuningResult, tune_rbf_kernel_residual
from option_gpr.kernels import RBFKernel
from option_gpr.metrics import mae, max_abs_error, mean_relative_error
from option_gpr.models import BlackScholesModel
from option_gpr.operators import BSLogOperator
from option_gpr.payoffs import call_boundary_values_log
from option_gpr.posterior import StackedOperatorGP


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Black-Scholes time and posterior-std experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    S_min: float = 1.0
    S_max: float = 200.0
    validation_S_min: float = 50.0
    validation_S_max: float = 150.0
    validation_S_step: float = 2.0
    m_values: tuple[int, ...] = (200, 400, 600, 800)
    initial_sigma_f: float = 100.0
    fixed_sigma_f: float | None = 1.0
    initial_ell_t: float = 0.8
    initial_ell_x: float = 1.2
    noise_int: float = 1e-2
    noise_bd: float = 1e-2
    jitter: float = 0.0
    maxiter: int = 300
    xatol: float = 1e-2
    fatol: float = 1e-2
    base_seed: int = 20260511
    output_dir: Path = Path("results/black_scholes/time_std")
    grid_sampling: str = "price_uniform"
    schema_version: str = "1"


@dataclass(frozen=True)
class NestedGridPool:
    """Largest nested random grid for one maturity."""

    X_int: NDArray[np.float64]
    X_terminal: NDArray[np.float64]
    X_lower: NDArray[np.float64]
    X_upper: NDArray[np.float64]


def split_boundary_count(d: int) -> tuple[int, int, int]:
    """Split total boundary points into terminal, lower, and upper counts."""

    if d < 3:
        raise ValueError(f"d must be at least 3, got {d!r}.")
    q, rem = divmod(d, 3)
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


def make_nested_grid_pool(
    config: ExperimentConfig,
    maturity: float,
    max_m: int,
    root_seed: int,
) -> NestedGridPool:
    """Generate largest nested train or tune pool for one maturity."""

    n_terminal, n_lower, n_upper = _max_boundary_counts(config)
    generators = _grid_point_generators(config.grid_sampling)
    rng = np.random.default_rng(root_seed)
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


def run_single_case(
    config: ExperimentConfig,
    maturity: float,
    m: int,
    train_grid: GridSet,
    tune_grid: GridSet,
    train_seed: int,
    tune_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, NDArray[np.float64]],
]:
    """Run one maturity and one fixed-ratio ``m=d`` configuration."""

    d = m
    n_terminal, n_lower, n_upper = split_boundary_count(d)
    model = BlackScholesModel(
        r=config.r,
        sigma=config.sigma,
        strike=config.strike,
        maturity=maturity,
    )

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

    fit_start = perf_counter()
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
    final_fit_time = perf_counter() - fit_start

    S0 = validation_spots(config)
    x0 = np.log(S0)
    X_star = np.column_stack([np.zeros_like(S0), x0])
    pricing_start = perf_counter()
    pred = gp.predict(X_star)
    pricing_time = perf_counter() - pricing_start

    std_start = perf_counter()
    _, var = gp.predict(X_star, return_var=True)
    posterior_std = np.sqrt(np.maximum(var, 0.0))
    posterior_std_time = perf_counter() - std_start

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
    total_time = (
        optimization_time
        + final_fit_time
        + pricing_time
        + posterior_std_time
        + benchmark_time
    )

    maturity_row = {
        "maturity": maturity,
        "m": m,
        "d": d,
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
        "benchmark_time_sec": benchmark_time,
        "total_time_sec": total_time,
        "tuning_success": tuning.success,
        "tuning_message": tuning.message,
        "tuning_nit": tuning.nit,
        "tuning_nfev": tuning.nfev,
        "tuning_objective_value": tuning.objective_value,
    }
    hyperparameter_row = _hyperparameter_row(maturity, m, d, tuning, config)
    prediction_rows = _prediction_rows(
        maturity=maturity,
        m=m,
        d=d,
        S0=S0,
        x0=x0,
        pred=pred,
        ref=ref,
        posterior_std=posterior_std,
        strike=config.strike,
    )
    arrays = {"S0": S0, "pred": pred, "ref": ref, "posterior_std": posterior_std}
    return maturity_row, hyperparameter_row, prediction_rows, arrays


def run_experiment(
    config: ExperimentConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run all configured Black-Scholes time/std experiment cases."""

    aggregate_inputs: dict[int, dict[str, Any]] = {}
    maturity_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    max_m = max(config.m_values)

    for maturity_index, maturity in enumerate(config.maturities):
        train_seed = config.base_seed + 100 * maturity_index + 1
        tune_seed = config.base_seed + 100 * maturity_index + 2
        train_pool = make_nested_grid_pool(config, maturity, max_m, train_seed)
        tune_pool = make_nested_grid_pool(config, maturity, max_m, tune_seed)
        for m in config.m_values:
            train_grid = trim_nested_grid(config, maturity, train_pool, m)
            tune_grid = trim_nested_grid(config, maturity, tune_pool, m)
            maturity_row, hyper_row, pred_rows, arrays = run_single_case(
                config,
                maturity=maturity,
                m=m,
                train_grid=train_grid,
                tune_grid=tune_grid,
                train_seed=train_seed,
                tune_seed=tune_seed,
            )
            maturity_rows.append(maturity_row)
            hyperparameter_rows.append(hyper_row)
            prediction_rows.extend(pred_rows)
            bucket = aggregate_inputs.setdefault(
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

    aggregate_rows = [
        _aggregate_row(
            config=config,
            m=m,
            maturity_rows=bucket["rows"],
            predictions=np.concatenate(bucket["predictions"]),
            references=np.concatenate(bucket["references"]),
            spots=np.concatenate(bucket["spots"]),
            posterior_stds=np.concatenate(bucket["posterior_stds"]),
        )
        for m, bucket in aggregate_inputs.items()
    ]
    return aggregate_rows, maturity_rows, hyperparameter_rows, prediction_rows


def write_outputs(
    config: ExperimentConfig,
    aggregate_rows: list[dict[str, Any]],
    maturity_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    """Write experiment output CSV and JSON files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(config.output_dir / "maturity_metrics.csv", maturity_rows)
    _write_csv(config.output_dir / "hyperparameters.csv", hyperparameter_rows)
    _write_csv(config.output_dir / "predictions.csv", prediction_rows)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_json(config), handle, indent=2)


def main() -> None:
    """Run the full experiment and write outputs."""

    config = ExperimentConfig()
    aggregate_rows, maturity_rows, hyperparameter_rows, prediction_rows = run_experiment(
        config
    )
    write_outputs(
        config,
        aggregate_rows,
        maturity_rows,
        hyperparameter_rows,
        prediction_rows,
    )
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


def _collocation_pairs(config: ExperimentConfig) -> list[dict[str, int]]:
    return [{"m": m, "d": m} for m in config.m_values]


def _aggregate_row(
    *,
    config: ExperimentConfig,
    m: int,
    maturity_rows: list[dict[str, Any]],
    predictions: NDArray[np.float64],
    references: NDArray[np.float64],
    spots: NDArray[np.float64],
    posterior_stds: NDArray[np.float64],
) -> dict[str, Any]:
    d = m
    n_terminal, n_lower, n_upper = split_boundary_count(d)
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    objectives = np.array([row["tuning_objective_value"] for row in maturity_rows])
    successes = [bool(row["tuning_success"]) for row in maturity_rows]
    return {
        "m": m,
        "d": d,
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
        "benchmark_time_sec": sum(row["benchmark_time_sec"] for row in maturity_rows),
        "total_time_sec": sum(row["total_time_sec"] for row in maturity_rows),
        "tuning_success_count": sum(successes),
        "tuning_failure_count": len(successes) - sum(successes),
        "mean_objective_value": float(np.mean(objectives)),
        "max_objective_value": float(np.max(objectives)),
    }


def _hyperparameter_row(
    maturity: float,
    m: int,
    d: int,
    tuning: ResidualTuningResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "maturity": maturity,
        "m": m,
        "d": d,
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
    maturity: float,
    m: int,
    d: int,
    S0: NDArray[np.float64],
    x0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    posterior_std: NDArray[np.float64],
    strike: float,
) -> list[dict[str, Any]]:
    return [
        {
            "maturity": maturity,
            "m": m,
            "d": d,
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
    data["grid_design"] = "nested_random_price_fixed_ratio"
    data["grid_design_note"] = (
        "For each maturity, largest train/tune pools are generated once. "
        "Interior, terminal, lower, and upper blocks are sampled sequentially "
        "from one random generator, matching the nested collocation experiment."
    )
    data["grid_sampling_note"] = (
        "'price_uniform' samples uniformly in S and returns x = log(S). "
        "'log_price_uniform' samples uniformly directly in x = log(S)."
    )
    data["collocation_pairs"] = _collocation_pairs(config)
    data["seed_policy"] = (
        "train_seed = base_seed + 100 * maturity_index + 1; "
        "tune_seed = base_seed + 100 * maturity_index + 2"
    )
    data["total_time_definition"] = (
        "optimization_time_sec + final_fit_time_sec + pricing_time_sec + "
        "posterior_std_time_sec + benchmark_time_sec"
    )
    return data


if __name__ == "__main__":
    main()
