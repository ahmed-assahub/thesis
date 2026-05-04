"""Nested Black-Scholes collocation-size sensitivity experiment."""

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
    """Configuration for the nested Black-Scholes collocation experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    S_min: float = 1.0
    S_max: float = 200.0
    validation_S_min: float = 50.0
    validation_S_max: float = 150.0
    validation_S_step: float = 2.0
    n_int_values: tuple[int, ...] = (200, 400, 600, 800)
    n_bd_ratios: tuple[float, ...] = (0.6, 0.75, 1.0)
    initial_sigma_f: float = 100.0
    fixed_sigma_f: float | None = 1.0
    initial_ell_t: float = 0.8
    initial_ell_x: float = 1.2
    noise_int: float = 1e-2
    noise_bd: float = 1e-2
    jitter: float = 0.0
    maxiter: int = 300
    xatol: float = 1e-6
    fatol: float = 1e-6
    base_seed: int = 20260511
    output_dir: Path = Path("results/black_scholes/collocation_sensitivity_nested")
    schema_version: str = "1"


@dataclass(frozen=True)
class BoundaryBlocks:
    """Separate terminal, lower, and upper boundary point blocks."""

    X_terminal: NDArray[np.float64]
    X_lower: NDArray[np.float64]
    X_upper: NDArray[np.float64]


@dataclass(frozen=True)
class NestedGridPool:
    """Largest nested random grid blocks for one maturity and ratio family."""

    X_int: NDArray[np.float64]
    X_terminal: NDArray[np.float64]
    X_lower: NDArray[np.float64]
    X_upper: NDArray[np.float64]


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


def make_nested_grid_pool(
    config: ExperimentConfig,
    maturity: float,
    max_n_int: int,
    max_n_bd: int,
    seed: int,
    boundary_counts: tuple[int, int, int] | None = None,
) -> NestedGridPool:
    """Generate largest nested point blocks for one maturity and ratio family."""

    if boundary_counts is None:
        boundary_counts = split_boundary_count(max_n_bd)
    n_terminal, n_lower, n_upper = boundary_counts
    rng = np.random.default_rng(seed)
    return NestedGridPool(
        X_int=make_random_price_interior_points(
            max_n_int,
            t_min=0.0,
            t_max=maturity,
            S_min=config.S_min,
            S_max=config.S_max,
            maturity=maturity,
            rng=rng,
        ),
        X_terminal=make_random_price_terminal_boundary(
            n_terminal,
            maturity=maturity,
            S_min=config.S_min,
            S_max=config.S_max,
            rng=rng,
        ),
        X_lower=make_random_price_lower_boundary(
            n_lower,
            t_min=0.0,
            maturity=maturity,
            S_min=config.S_min,
            rng=rng,
        ),
        X_upper=make_random_price_upper_boundary(
            n_upper,
            t_min=0.0,
            maturity=maturity,
            S_max=config.S_max,
            rng=rng,
        ),
    )


def trim_boundary_blocks(pool: NestedGridPool, n_bd: int) -> BoundaryBlocks:
    """Return nested boundary block prefixes for the requested total size."""

    n_terminal, n_lower, n_upper = split_boundary_count(n_bd)
    return BoundaryBlocks(
        X_terminal=pool.X_terminal[:n_terminal],
        X_lower=pool.X_lower[:n_lower],
        X_upper=pool.X_upper[:n_upper],
    )


def trim_nested_grid(
    config: ExperimentConfig,
    maturity: float,
    pool: NestedGridPool,
    n_int: int,
    n_bd: int,
) -> GridSet:
    """Trim a largest nested pool down to one grid size."""

    blocks = trim_boundary_blocks(pool, n_bd)
    X_bd = combine_boundary_points(blocks.X_terminal, blocks.X_lower, blocks.X_upper)
    y_bd = make_boundary_value_fn(config, maturity)(X_bd)
    return GridSet(X_int=pool.X_int[:n_int], X_bd=X_bd, y_bd=y_bd)


def run_single_case(
    config: ExperimentConfig,
    maturity: float,
    n_int: int,
    n_bd: int,
    train_grid: GridSet,
    tune_grid: GridSet,
    train_seed: int,
    tune_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Run one maturity and nested collocation configuration."""

    n_terminal, n_lower, n_upper = split_boundary_count(n_bd)
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
    X_star = np.column_stack([np.zeros_like(S0), np.log(S0)])
    pricing_start = perf_counter()
    pred = gp.predict(X_star)
    pricing_time = perf_counter() - pricing_start
    ref = black_scholes_call_price_log(
        X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
    )
    metric_values = compute_metrics(S0, pred, ref, config.strike)
    total_time = optimization_time + final_fit_time + pricing_time

    maturity_row = {
        "maturity": maturity,
        "n_int": n_int,
        "n_bd": n_bd,
        "n_terminal": n_terminal,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "train_seed": train_seed,
        "tune_seed": tune_seed,
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "optimization_time_sec": optimization_time,
        "final_fit_time_sec": final_fit_time,
        "pricing_time_sec": pricing_time,
        "total_time_sec": total_time,
        "tuning_success": tuning.success,
        "tuning_message": tuning.message,
        "tuning_nit": tuning.nit,
        "tuning_nfev": tuning.nfev,
        "tuning_objective_value": tuning.objective_value,
    }
    hyperparameter_row = _hyperparameter_row(maturity, n_int, n_bd, tuning, config)
    arrays = {"S0": S0, "pred": pred, "ref": ref}
    return maturity_row, hyperparameter_row, _timing_row(maturity_row), arrays


def run_experiment(
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all nested Black-Scholes collocation sensitivity cases."""

    aggregate_inputs: dict[tuple[int, int], dict[str, Any]] = {}
    maturity_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    for ratio_index, ratio in enumerate(config.n_bd_ratios):
        max_n_int = max(config.n_int_values)
        max_n_bd = max(int(round(ratio * n_int)) for n_int in config.n_int_values)
        boundary_counts = _max_boundary_counts_for_ratio(config, ratio)
        for maturity_index, maturity in enumerate(config.maturities):
            train_seed = config.base_seed + 100 * maturity_index + 1
            tune_seed = config.base_seed  + 100 * maturity_index + 2
            train_pool = make_nested_grid_pool(
                config,
                maturity,
                max_n_int=max_n_int,
                max_n_bd=max_n_bd,
                seed=train_seed,
                boundary_counts=boundary_counts,
            )
            tune_pool = make_nested_grid_pool(
                config,
                maturity,
                max_n_int=max_n_int,
                max_n_bd=max_n_bd,
                seed=tune_seed,
                boundary_counts=boundary_counts,
            )
            for n_int in config.n_int_values:
                n_bd = int(round(ratio * n_int))
                train_grid = trim_nested_grid(config, maturity, train_pool, n_int, n_bd)
                tune_grid = trim_nested_grid(config, maturity, tune_pool, n_int, n_bd)
                maturity_row, hyper_row, _, arrays = run_single_case(
                    config,
                    maturity=maturity,
                    n_int=n_int,
                    n_bd=n_bd,
                    train_grid=train_grid,
                    tune_grid=tune_grid,
                    train_seed=train_seed,
                    tune_seed=tune_seed,
                )
                maturity_rows.append(maturity_row)
                hyperparameter_rows.append(hyper_row)
                key = (n_int, n_bd)
                bucket = aggregate_inputs.setdefault(
                    key,
                    {"rows": [], "predictions": [], "references": [], "spots": []},
                )
                bucket["rows"].append(maturity_row)
                bucket["predictions"].append(arrays["pred"])
                bucket["references"].append(arrays["ref"])
                bucket["spots"].append(arrays["S0"])

    aggregate_rows = [
        _aggregate_row(
            config=config,
            n_int=n_int,
            n_bd=n_bd,
            maturity_rows=bucket["rows"],
            predictions=np.concatenate(bucket["predictions"]),
            references=np.concatenate(bucket["references"]),
            spots=np.concatenate(bucket["spots"]),
        )
        for (n_int, n_bd), bucket in aggregate_inputs.items()
    ]
    return aggregate_rows, maturity_rows, hyperparameter_rows


def write_outputs(
    config: ExperimentConfig,
    aggregate_rows: list[dict[str, Any]],
    maturity_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
) -> None:
    """Write experiment output CSV and JSON files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(config.output_dir / "maturity_metrics.csv", maturity_rows)
    _write_csv(config.output_dir / "hyperparameters.csv", hyperparameter_rows)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_json(config), handle, indent=2)


def main() -> None:
    """Run the full nested experiment and write outputs."""

    config = ExperimentConfig()
    aggregate_rows, maturity_rows, hyperparameter_rows = run_experiment(config)
    write_outputs(config, aggregate_rows, maturity_rows, hyperparameter_rows)
    print(f"Wrote results to {config.output_dir}")


def validation_spots(config: ExperimentConfig) -> np.ndarray:
    """Return validation spots in ordinary price coordinates."""

    return np.arange(
        config.validation_S_min,
        config.validation_S_max + 1e-12,
        config.validation_S_step,
        dtype=float,
    )


def compute_metrics(
    S0: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
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


def _max_boundary_counts_for_ratio(
    config: ExperimentConfig, ratio: float
) -> tuple[int, int, int]:
    counts = [
        split_boundary_count(int(round(ratio * n_int)))
        for n_int in config.n_int_values
    ]
    return tuple(max(parts) for parts in zip(*counts, strict=True))


def _collocation_pairs(config: ExperimentConfig) -> list[tuple[int, int]]:
    return [
        (n_int, int(round(ratio * n_int)))
        for ratio in config.n_bd_ratios
        for n_int in config.n_int_values
    ]


def _aggregate_row(
    *,
    config: ExperimentConfig,
    n_int: int,
    n_bd: int,
    maturity_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    references: np.ndarray,
    spots: np.ndarray,
) -> dict[str, Any]:
    n_terminal, n_lower, n_upper = split_boundary_count(n_bd)
    metric_values = compute_metrics(spots, predictions, references, config.strike)
    objectives = np.array([row["tuning_objective_value"] for row in maturity_rows])
    successes = [bool(row["tuning_success"]) for row in maturity_rows]
    return {
        "n_int": n_int,
        "n_bd": n_bd,
        "n_terminal": n_terminal,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "n_maturities": len(config.maturities),
        "n_validation_spots": validation_spots(config).shape[0],
        "mae": metric_values["mae"],
        "max_abs_error": metric_values["max_abs_error"],
        "mre_itm": metric_values["mre_itm"],
        "optimization_time_sec": sum(row["optimization_time_sec"] for row in maturity_rows),
        "final_fit_time_sec": sum(row["final_fit_time_sec"] for row in maturity_rows),
        "pricing_time_sec": sum(row["pricing_time_sec"] for row in maturity_rows),
        "total_time_sec": sum(row["total_time_sec"] for row in maturity_rows),
        "tuning_success_count": sum(successes),
        "tuning_failure_count": len(successes) - sum(successes),
        "mean_objective_value": float(np.mean(objectives)),
        "max_objective_value": float(np.max(objectives)),
    }


def _hyperparameter_row(
    maturity: float,
    n_int: int,
    n_bd: int,
    tuning: ResidualTuningResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "maturity": maturity,
        "n_int": n_int,
        "n_bd": n_bd,
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


def _timing_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "optimization_time_sec": row["optimization_time_sec"],
        "final_fit_time_sec": row["final_fit_time_sec"],
        "pricing_time_sec": row["pricing_time_sec"],
        "total_time_sec": row["total_time_sec"],
    }


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
    data["grid_design"] = "nested_random_price"
    data["grid_design_note"] = (
        "For each maturity and boundary ratio, largest train/tune pools are "
        "generated once. Smaller grids are prefixes of the corresponding "
        "interior, terminal, lower, and upper blocks."
    )
    data["collocation_pairs"] = [
        {"n_int": n_int, "n_bd": n_bd} for n_int, n_bd in _collocation_pairs(config)
    ]
    data["seed_policy"] = (
        "train_seed = base_seed + 10000 * ratio_index + 100 * maturity_index + 1; "
        "tune_seed = base_seed + 10000 * ratio_index + 100 * maturity_index + 2"
    )
    return data


if __name__ == "__main__":
    main()
