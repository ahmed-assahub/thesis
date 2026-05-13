"""Black-Scholes GPR Greek accuracy experiment."""

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
    black_scholes_call_greeks_log,
    black_scholes_call_price_log,
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
from option_gpr.models import BlackScholesModel  # noqa: E402
from option_gpr.operators import BSLogOperator  # noqa: E402
from option_gpr.payoffs import call_boundary_values_log  # noqa: E402
from option_gpr.posterior import StackedOperatorGP  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Black-Scholes GPR Greek experiment."""

    r: float = 0.05
    sigma: float = 0.25
    strike: float = 100.0
    maturities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    S_min: float = 1.0
    S_max: float = 200.0
    validation_S_min: float = 50.0
    validation_S_max: float = 150.0
    validation_S_step: float = 2.0
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
    xatol: float = 1e-6
    fatol: float = 1e-6
    base_seed: int = 20260511
    output_dir: Path = Path("results/black_scholes/greeks")
    grid_sampling: str = "price_uniform"
    schema_version: str = "1"


@dataclass(frozen=True)
class NestedGridPool:
    """Largest train or tune grid blocks for one maturity."""

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
    """Return Black-Scholes call boundary values."""

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
    seed: int,
) -> NestedGridPool:
    """Generate fixed-size train or tune grid blocks for one maturity."""

    n_terminal, n_lower, n_upper = split_boundary_count(config.n_bd)
    generators = _grid_point_generators(config.grid_sampling)
    rng = np.random.default_rng(seed)
    return NestedGridPool(
        X_int=generators["interior"](
            config.n_int,
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
) -> GridSet:
    """Build the configured fixed grid from a generated block pool."""

    X_bd = combine_boundary_points(pool.X_terminal, pool.X_lower, pool.X_upper)
    y_bd = make_boundary_value_fn(config, maturity)(X_bd)
    return GridSet(X_int=pool.X_int, X_bd=X_bd, y_bd=y_bd)


def run_single_case(
    config: ExperimentConfig,
    maturity: float,
    train_grid: GridSet,
    tune_grid: GridSet,
    train_seed: int,
    tune_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, NDArray[np.float64]]]:
    """Run one maturity case."""

    n_terminal, n_lower, n_upper = split_boundary_count(config.n_bd)
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
    price_gpr = gp.predict(X_star)
    pricing_time = perf_counter() - pricing_start

    greek_start = perf_counter()
    delta_gpr, gamma_gpr, theta_gpr = gp.predict_greeks(X_star)
    greek_time = perf_counter() - greek_start

    reference_start = perf_counter()
    price_ref = black_scholes_call_price_log(
        X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
    )
    delta_ref, gamma_ref, theta_ref = black_scholes_call_greeks_log(
        X_star,
        strike=config.strike,
        maturity=maturity,
        r=config.r,
        sigma=config.sigma,
    )
    reference_time = perf_counter() - reference_start
    _validate_finite_arrays(
        price_gpr,
        price_ref,
        delta_gpr,
        delta_ref,
        gamma_gpr,
        gamma_ref,
        theta_gpr,
        theta_ref,
    )

    metrics = compute_all_metrics(
        S0=S0,
        strike=config.strike,
        price_gpr=price_gpr,
        price_ref=price_ref,
        delta_gpr=delta_gpr,
        delta_ref=delta_ref,
        gamma_gpr=gamma_gpr,
        gamma_ref=gamma_ref,
        theta_gpr=theta_gpr,
        theta_ref=theta_ref,
    )
    total_gpr_time = optimization_time + final_fit_time + pricing_time + greek_time

    maturity_row = {
        "model": "black_scholes",
        "maturity": maturity,
        "n_int": config.n_int,
        "n_bd": config.n_bd,
        "n_terminal": n_terminal,
        "n_lower": n_lower,
        "n_upper": n_upper,
        "train_seed": train_seed,
        "tune_seed": tune_seed,
        **metrics,
        "optimization_time_sec": optimization_time,
        "final_fit_time_sec": final_fit_time,
        "pricing_time_sec": pricing_time,
        "greek_time_sec": greek_time,
        "reference_time_sec": reference_time,
        "total_gpr_time_sec": total_gpr_time,
        "tuning_success": tuning.success,
        "tuning_message": tuning.message,
        "tuning_nit": tuning.nit,
        "tuning_nfev": tuning.nfev,
        "tuning_objective_value": tuning.objective_value,
    }
    arrays = {
        "S0": S0,
        "price_gpr": price_gpr,
        "price_ref": price_ref,
        "delta_gpr": delta_gpr,
        "delta_ref": delta_ref,
        "gamma_gpr": gamma_gpr,
        "gamma_ref": gamma_ref,
        "theta_gpr": theta_gpr,
        "theta_ref": theta_ref,
    }
    prediction_rows = _prediction_rows(
        model_name="black_scholes",
        maturity=maturity,
        S0=S0,
        x0=x0,
        strike=config.strike,
        arrays=arrays,
    )
    return maturity_row, _hyperparameter_row(maturity, tuning, config), prediction_rows, arrays


def run_experiment(
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all configured Black-Scholes Greek cases."""

    maturity_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate_arrays: dict[str, list[NDArray[np.float64]]] = {
        "S0": [],
        "price_gpr": [],
        "price_ref": [],
        "delta_gpr": [],
        "delta_ref": [],
        "gamma_gpr": [],
        "gamma_ref": [],
        "theta_gpr": [],
        "theta_ref": [],
    }

    for maturity_index, maturity in enumerate(config.maturities):
        train_seed = config.base_seed + 100 * maturity_index + 1
        tune_seed = config.base_seed + 100 * maturity_index + 2
        train_pool = make_nested_grid_pool(config, maturity, train_seed)
        tune_pool = make_nested_grid_pool(config, maturity, tune_seed)
        train_grid = trim_nested_grid(config, maturity, train_pool)
        tune_grid = trim_nested_grid(config, maturity, tune_pool)
        maturity_row, hyper_row, pred_rows, arrays = run_single_case(
            config,
            maturity=maturity,
            train_grid=train_grid,
            tune_grid=tune_grid,
            train_seed=train_seed,
            tune_seed=tune_seed,
        )
        maturity_rows.append(maturity_row)
        hyperparameter_rows.append(hyper_row)
        prediction_rows.extend(pred_rows)
        for key in aggregate_arrays:
            aggregate_arrays[key].append(arrays[key])

    aggregate_row = _aggregate_row(
        config=config,
        maturity_rows=maturity_rows,
        arrays={key: np.concatenate(value) for key, value in aggregate_arrays.items()},
    )
    return [aggregate_row], maturity_rows, prediction_rows, hyperparameter_rows


def write_outputs(
    config: ExperimentConfig,
    aggregate_rows: list[dict[str, Any]],
    maturity_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
) -> None:
    """Write experiment output CSV and JSON files."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(config.output_dir / "maturity_metrics.csv", maturity_rows)
    _write_csv(config.output_dir / "predictions.csv", prediction_rows)
    _write_csv(config.output_dir / "hyperparameters.csv", hyperparameter_rows)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_json(config), handle, indent=2)


def main() -> None:
    """Run the full experiment and write outputs."""

    config = ExperimentConfig()
    rows = run_experiment(config)
    write_outputs(config, *rows)
    summary = rows[0][0]
    print(f"Wrote results to {config.output_dir}")
    print(
        "Aggregate metrics: "
        f"price_mae={summary['price_mae']:.6g}, "
        f"delta_mae={summary['delta_mae']:.6g}, "
        f"gamma_mae={summary['gamma_mae']:.6g}, "
        f"theta_mae={summary['theta_mae']:.6g}"
    )


def validation_spots(config: ExperimentConfig) -> NDArray[np.float64]:
    """Return validation spots in ordinary price coordinates."""

    return np.arange(
        config.validation_S_min,
        config.validation_S_max + 1e-12,
        config.validation_S_step,
        dtype=float,
    )


def compute_price_metrics(
    S0: NDArray[np.float64],
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    strike: float,
) -> dict[str, float]:
    """Return price MAE, MaxAE, and ITM mean relative error."""

    itm = S0 > strike
    if not np.any(itm):
        raise ValueError("At least one validation spot must be in the money.")
    return {
        "price_mae": mae(pred, ref),
        "price_max_abs_error": max_abs_error(pred, ref),
        "price_mre_itm": mean_relative_error(pred[itm], ref[itm]),
    }


def compute_greek_metrics(
    prefix: str,
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    *,
    include_relative_diagnostics: bool = True,
) -> dict[str, float]:
    """Return Greek MAE and MaxAE, optionally with maturity-relative diagnostics."""

    greek_mae = mae(pred, ref)
    metrics = {
        f"{prefix}_mae": greek_mae,
        f"{prefix}_max_abs_error": max_abs_error(pred, ref),
    }
    if include_relative_diagnostics:
        mean_abs_reference = float(np.mean(np.abs(ref)))
        if not np.isfinite(mean_abs_reference) or mean_abs_reference <= 0.0:
            raise ValueError(
                f"{prefix}_mean_abs_reference must be positive and finite."
            )
        metrics[f"{prefix}_mean_abs_reference"] = mean_abs_reference
        metrics[f"{prefix}_rel_mae"] = float(greek_mae / mean_abs_reference)
    return metrics


def compute_all_metrics(
    *,
    S0: NDArray[np.float64],
    strike: float,
    price_gpr: NDArray[np.float64],
    price_ref: NDArray[np.float64],
    delta_gpr: NDArray[np.float64],
    delta_ref: NDArray[np.float64],
    gamma_gpr: NDArray[np.float64],
    gamma_ref: NDArray[np.float64],
    theta_gpr: NDArray[np.float64],
    theta_ref: NDArray[np.float64],
    include_relative_greek_diagnostics: bool = True,
) -> dict[str, float]:
    """Return price and Greek error metrics."""

    return {
        **compute_price_metrics(S0, price_gpr, price_ref, strike),
        **compute_greek_metrics(
            "delta",
            delta_gpr,
            delta_ref,
            include_relative_diagnostics=include_relative_greek_diagnostics,
        ),
        **compute_greek_metrics(
            "gamma",
            gamma_gpr,
            gamma_ref,
            include_relative_diagnostics=include_relative_greek_diagnostics,
        ),
        **compute_greek_metrics(
            "theta",
            theta_gpr,
            theta_ref,
            include_relative_diagnostics=include_relative_greek_diagnostics,
        ),
    }


def _make_bs_operator(model: BlackScholesModel, kernel: RBFKernel) -> BSLogOperator:
    return BSLogOperator(model=model, kernel=kernel)


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


def _aggregate_row(
    *,
    config: ExperimentConfig,
    maturity_rows: list[dict[str, Any]],
    arrays: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    metrics = compute_all_metrics(
        S0=arrays["S0"],
        strike=config.strike,
        price_gpr=arrays["price_gpr"],
        price_ref=arrays["price_ref"],
        delta_gpr=arrays["delta_gpr"],
        delta_ref=arrays["delta_ref"],
        gamma_gpr=arrays["gamma_gpr"],
        gamma_ref=arrays["gamma_ref"],
        theta_gpr=arrays["theta_gpr"],
        theta_ref=arrays["theta_ref"],
        include_relative_greek_diagnostics=False,
    )
    return {
        "model": "black_scholes",
        "n_int": config.n_int,
        "n_bd": config.n_bd,
        "n_maturities": len(config.maturities),
        "n_validation_spots": validation_spots(config).shape[0],
        **metrics,
        "optimization_time_sec": sum(row["optimization_time_sec"] for row in maturity_rows),
        "final_fit_time_sec": sum(row["final_fit_time_sec"] for row in maturity_rows),
        "pricing_time_sec": sum(row["pricing_time_sec"] for row in maturity_rows),
        "greek_time_sec": sum(row["greek_time_sec"] for row in maturity_rows),
        "reference_time_sec": sum(row["reference_time_sec"] for row in maturity_rows),
        "total_gpr_time_sec": sum(row["total_gpr_time_sec"] for row in maturity_rows),
    }


def _prediction_rows(
    *,
    model_name: str,
    maturity: float,
    S0: NDArray[np.float64],
    x0: NDArray[np.float64],
    strike: float,
    arrays: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (S, x) in enumerate(zip(S0, x0, strict=True)):
        row = {
            "model": model_name,
            "maturity": maturity,
            "S0": float(S),
            "x0": float(x),
            "is_itm": bool(S > strike),
        }
        for name in ("price", "delta", "gamma", "theta"):
            gpr = float(arrays[f"{name}_gpr"][i])
            ref = float(arrays[f"{name}_ref"][i])
            row[f"{name}_gpr"] = gpr
            row[f"{name}_reference"] = ref
            row[f"{name}_error"] = gpr - ref
            row[f"{name}_abs_error"] = abs(gpr - ref)
        rows.append(row)
    return rows


def _hyperparameter_row(
    maturity: float,
    tuning: ResidualTuningResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
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


def _validate_finite_arrays(*arrays: NDArray[np.float64]) -> None:
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("price and Greek predictions/references must be finite.")


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
    data["grid_design"] = "fixed_nested_style_random_price"
    data["grid_sampling_note"] = (
        "'price_uniform' samples uniformly in S and returns x = log(S). "
        "'log_price_uniform' samples uniformly directly in x = log(S)."
    )
    data["seed_policy"] = (
        "train_seed = base_seed + 100 * maturity_index + 1; "
        "tune_seed = base_seed + 100 * maturity_index + 2"
    )
    data["greek_convention_note"] = (
        "GPR Greeks are computed analytically with gp.predict_greeks. Delta "
        "and Gamma are price-coordinate Greeks. Theta is dV/dt at fixed "
        "maturity. Reference Greeks are Black-Scholes closed forms. No finite "
        "differences are used in experiment computations."
    )
    return data


if __name__ == "__main__":
    main()
