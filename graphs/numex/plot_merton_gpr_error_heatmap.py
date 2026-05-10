import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

REQUIRED_COLUMNS = {
    "method",
    "m",
    "n_int",
    "n_bd",
    "maturity",
    "S0",
    "x0",
    "prediction",
    "reference_series",
    "spread",
    "abs_error",
    "is_itm",
    "posterior_std",
}

HEATMAP_COLORS = ["#8fb78f", "#4b9a91", "#1f6f83", "#2b2d70"]


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


def parse_numeric_row(row):
    return {
        "maturity": float(row["maturity"]),
        "m": int(float(row["m"])),
        "n_bd": int(float(row.get("n_bd", row.get("d")))),
        "S0": float(row["S0"]),
        "abs_error": float(row["abs_error"]),
    }


def load_error_grid(input_path, m, d):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        if "n_bd" not in fieldnames and "d" in fieldnames:
            fieldnames.add("n_bd")
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                "Prediction CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        rows = [
            parsed
            for row in reader
            for parsed in [parse_numeric_row(row)]
            if parsed["m"] == m and parsed["n_bd"] == d
        ]

    if not rows:
        raise ValueError(f"No prediction rows found for m={m}, d={d}")

    maturities = np.array(sorted({row["maturity"] for row in rows}), dtype=float)
    spots = np.array(sorted({row["S0"] for row in rows}), dtype=float)
    maturity_index = {value: index for index, value in enumerate(maturities)}
    spot_index = {value: index for index, value in enumerate(spots)}

    errors = np.full((maturities.size, spots.size), np.nan, dtype=float)
    for row in rows:
        row_index = maturity_index[row["maturity"]]
        column_index = spot_index[row["S0"]]
        if np.isfinite(errors[row_index, column_index]):
            raise ValueError(
                "Duplicate prediction row for "
                f"maturity={row['maturity']}, S0={row['S0']}, m={m}, d={d}"
            )
        errors[row_index, column_index] = row["abs_error"]

    if not np.all(np.isfinite(errors)):
        raise ValueError(f"Incomplete prediction grid for m={m}, d={d}")

    return maturities, spots, errors


def plot_error_heatmap(
    maturities,
    spots,
    errors,
    output_path,
    *,
    strike=100.0,
    show_strike_line=True,
):
    plt.rcParams.update(PLOT_STYLE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colormap = LinearSegmentedColormap.from_list(
        "teal_blue_error",
        HEATMAP_COLORS,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    style_axis(ax)

    spot_edges = coordinate_edges(spots)
    maturity_edges = coordinate_edges(maturities)
    mesh = ax.pcolormesh(
        spot_edges,
        maturity_edges,
        errors,
        cmap=colormap,
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
    ax.set_ylabel(r"Maturity $T$", fontsize=8)
    ax.set_xticks(spots[:: max(1, spots.size // 10)])
    ax.set_yticks(maturities)

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.ax.tick_params(labelsize=9, width=0.6, length=3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    default_input = Path("results/merton_jump_diffusion/grid_mc_comparison/gpr_predictions.csv")
    default_output = Path(__file__).with_suffix(".pdf")

    parser = argparse.ArgumentParser(
        description="Plot a Merton GPR pointwise absolute-error heatmap."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--no-strike-line", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    maturities, spots, errors = load_error_grid(args.input, args.m, args.d)

    plot_error_heatmap(
        maturities,
        spots,
        errors,
        args.output,
        show_strike_line=not args.no_strike_line,
    )
    print(f"Saved {args.output}")

    if args.png:
        png_output = args.output.with_suffix(".png")
        plot_error_heatmap(
            maturities,
            spots,
            errors,
            png_output,
            show_strike_line=not args.no_strike_line,
        )
        print(f"Saved {png_output}")


if __name__ == "__main__":
    main()
