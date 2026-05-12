import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

SINGLE_REQUIRED_COLUMNS = {
    "remaining_maturity",
    "S0",
    "abs_error",
}

MATURITY_REQUIRED_COLUMNS = {
    "maturity",
    "S0",
    "abs_error",
}

DELTA_HEATMAP_COLORS = ["#8fb78f", "#f7f7f2", "#2b2d70"]


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step])

    midpoints = 0.5 * (values[1:] + values[:-1])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def parse_single_row(row):
    return (
        float(row["remaining_maturity"]),
        float(row["S0"]),
        float(row["abs_error"]),
    )


def parse_maturity_row(row):
    return (
        float(row["maturity"]),
        float(row["S0"]),
        float(row["abs_error"]),
    )


def load_single_surface_errors(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Single-surface prediction CSV not found: {input_path}")

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(SINGLE_REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                "Single-surface prediction CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        return _rows_to_error_map(
            [parse_single_row(row) for row in reader],
            source_name="single-surface",
        )


def load_maturity_wise_errors(input_path, grid_size):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Maturity-wise prediction CSV not found: {input_path}")

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(MATURITY_REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                "Maturity-wise prediction CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        rows = [
            parse_maturity_row(row)
            for row in reader
            if _matches_grid_size(row, grid_size, fieldnames)
        ]

    if not rows:
        raise ValueError(f"No maturity-wise prediction rows found for grid_size={grid_size}")

    return _rows_to_error_map(rows, source_name="maturity-wise")


def _matches_grid_size(row, grid_size, fieldnames):
    if {"n_int", "n_bd"}.issubset(fieldnames):
        return (
            int(float(row["n_int"])) == grid_size
            and int(float(row["n_bd"])) == grid_size
        )
    if "m" in fieldnames:
        return int(float(row["m"])) == grid_size
    raise ValueError(
        "Maturity-wise prediction CSV must contain either n_int/n_bd or m "
        "for grid-size filtering."
    )


def _rows_to_error_map(rows, *, source_name):
    errors = {}
    for maturity, spot, abs_error in rows:
        key = (maturity, spot)
        if key in errors:
            raise ValueError(
                f"Duplicate {source_name} prediction row for maturity={maturity}, "
                f"S0={spot}"
            )
        errors[key] = abs_error
    if not errors:
        raise ValueError(f"No {source_name} prediction rows found.")
    return errors


def load_delta_grid(single_input, maturity_input, grid_size):
    single_errors = load_single_surface_errors(single_input)
    maturity_errors = load_maturity_wise_errors(maturity_input, grid_size)

    single_keys = set(single_errors)
    maturity_keys = set(maturity_errors)
    if single_keys != maturity_keys:
        missing_from_maturity = sorted(single_keys - maturity_keys)[:5]
        missing_from_single = sorted(maturity_keys - single_keys)[:5]
        raise ValueError(
            "Single-surface and maturity-wise prediction grids do not match. "
            f"Missing from maturity-wise sample: {missing_from_maturity}; "
            f"missing from single-surface sample: {missing_from_single}."
        )

    maturities = np.array(sorted({key[0] for key in single_keys}), dtype=float)
    spots = np.array(sorted({key[1] for key in single_keys}), dtype=float)
    maturity_index = {value: index for index, value in enumerate(maturities)}
    spot_index = {value: index for index, value in enumerate(spots)}

    deltas = np.full((maturities.size, spots.size), np.nan, dtype=float)
    for key in single_keys:
        maturity, spot = key
        row_index = maturity_index[maturity]
        column_index = spot_index[spot]
        deltas[row_index, column_index] = (
            single_errors[key] - maturity_errors[key]
        )

    if not np.all(np.isfinite(deltas)):
        raise ValueError("Incomplete delta grid after matching prediction rows.")

    return maturities, spots, deltas


def plot_delta_heatmap(
    maturities,
    spots,
    deltas,
    output_path,
    *,
    strike=100.0,
    show_strike_line=True,
):
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    style_axis(ax)

    max_abs_delta = float(np.max(np.abs(deltas)))
    if max_abs_delta == 0.0:
        max_abs_delta = 1e-12
    norm = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)
    colormap = LinearSegmentedColormap.from_list(
        "muted_blue_green_delta",
        DELTA_HEATMAP_COLORS,
    )

    spot_edges = coordinate_edges(spots)
    maturity_edges = coordinate_edges(maturities)
    mesh = ax.pcolormesh(
        spot_edges,
        maturity_edges,
        deltas,
        cmap=colormap,
        norm=norm,
        shading="auto",
    )

    if show_strike_line and spots.min() <= strike <= spots.max():
        ax.axvline(
            strike,
            color="black",
            linewidth=0.7,
            linestyle="dashed",
            alpha=0.75,
        )

    ax.set_xlabel(r"Initial asset price $S_0$", fontsize=8)
    ax.set_ylabel(r"Remaining maturity $T$", fontsize=8)
    ax.set_xticks(spots[:: max(1, spots.size // 10)])
    ax.set_yticks(maturities)

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Difference in absolute error", fontsize=8)
    colorbar.ax.tick_params(labelsize=9, width=0.6, length=3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    default_single_input = Path(
        "results/merton_jump_diffusion/single_surface_mc_comparison/gpr_predictions.csv"
    )
    default_maturity_input = Path(
        "results/merton_jump_diffusion/grid_mc_comparison/gpr_predictions.csv"
    )
    default_output = Path(__file__).with_suffix(".pdf")

    parser = argparse.ArgumentParser(
        description=(
            "Plot single-surface minus maturity-wise Merton GPR absolute-error "
            "differences."
        )
    )
    parser.add_argument("--single-input", type=Path, default=default_single_input)
    parser.add_argument("--maturity-input", type=Path, default=default_maturity_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--no-strike-line", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    maturities, spots, deltas = load_delta_grid(
        args.single_input,
        args.maturity_input,
        args.grid_size,
    )

    plot_delta_heatmap(
        maturities,
        spots,
        deltas,
        args.output,
        strike=args.strike,
        show_strike_line=not args.no_strike_line,
    )
    print(f"Saved {args.output}")

    if args.png:
        png_output = args.output.with_suffix(".png")
        plot_delta_heatmap(
            maturities,
            spots,
            deltas,
            png_output,
            strike=args.strike,
            show_strike_line=not args.no_strike_line,
        )
        print(f"Saved {png_output}")


if __name__ == "__main__":
    main()
