"""Plot Merton jump-diffusion Greek references against GPR approximations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


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
    "maturity",
    "S0",
    "delta_gpr",
    "delta_reference",
    "gamma_gpr",
    "gamma_reference",
    "theta_gpr",
    "theta_reference",
}

GREEKS = (
    ("delta", r"$\Delta$"),
    ("gamma", r"$\Gamma$"),
    ("theta", r"$\Theta$"),
)
SELECTED_MATURITIES = (0.5, 1.5, 3.0)
DEFAULT_MATURITIES = (0.5,)
REFERENCE_COLOR = "#274c5e"
GPR_COLOR = "#d08a3e"
FIGURE_DPI = 150


def style_axis(ax: Any) -> None:
    """Apply the numerical-experiment axis style."""

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)


def parse_numeric_row(row: dict[str, str]) -> dict[str, float]:
    """Parse only the numeric columns needed for plotting."""

    parsed = {
        "maturity": float(row["maturity"]),
        "S0": float(row["S0"]),
    }
    for greek, _ in GREEKS:
        parsed[f"{greek}_gpr"] = float(row[f"{greek}_gpr"])
        parsed[f"{greek}_reference"] = float(row[f"{greek}_reference"])
    return parsed


def load_prediction_rows(
    input_path: Path,
    selected_maturities: tuple[float, ...] = SELECTED_MATURITIES,
) -> dict[float, list[dict[str, float]]]:
    """Load and group prediction rows by maturity."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")

    grouped: dict[float, list[dict[str, float]]] = {}
    seen: set[tuple[float, float]] = set()
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                "Prediction CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            parsed = parse_numeric_row(row)
            key = (parsed["maturity"], parsed["S0"])
            if key in seen:
                raise ValueError(
                    "Duplicate prediction row for "
                    f"maturity={parsed['maturity']}, S0={parsed['S0']}"
                )
            seen.add(key)
            grouped.setdefault(parsed["maturity"], []).append(parsed)

    if not grouped:
        raise ValueError(f"Prediction CSV contains no rows: {input_path}")

    for maturity in selected_maturities:
        if maturity not in grouped:
            raise ValueError(f"Selected maturity T={maturity:g} is absent.")
        grouped[maturity].sort(key=lambda item: item["S0"])

    return grouped


def plot_greeks(
    grouped: dict[float, list[dict[str, float]]],
    output_path: Path,
    *,
    selected_maturities: tuple[float, ...] = SELECTED_MATURITIES,
    strike: float = 100.0,
    show_row_labels: bool = True,
) -> None:
    """Create the Merton Greek comparison figure."""

    plt.rcParams.update(PLOT_STYLE)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(selected_maturities) == 1:
        _plot_single_maturity_figure5_4(
            grouped,
            output_path,
            maturity=selected_maturities[0],
            strike=strike,
        )
        return

    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(selected_maturities),
        figsize=_figure_size(len(selected_maturities)),
        dpi=FIGURE_DPI,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    fig.subplots_adjust(
        left=0.11 if len(selected_maturities) == 1 else 0.095,
        right=0.985,
        top=0.945,
        bottom=0.12 if len(selected_maturities) == 1 else 0.085,
        wspace=0.18,
        hspace=0.22 if len(selected_maturities) == 1 else 0.18,
    )

    reference_handle = None
    gpr_handle = None
    for row_index, (greek, label) in enumerate(GREEKS):
        for column_index, maturity in enumerate(selected_maturities):
            ax = axes[row_index, column_index]
            style_axis(ax)
            rows = grouped[maturity]
            spots = np.array([row["S0"] for row in rows], dtype=float)
            reference = np.array(
                [row[f"{greek}_reference"] for row in rows], dtype=float
            )
            gpr = np.array([row[f"{greek}_gpr"] for row in rows], dtype=float)

            (reference_handle,) = ax.plot(
                spots,
                reference,
                color=REFERENCE_COLOR,
                linewidth=1.2,
                label="Reference",
                zorder=4,
            )
            gpr_handle = ax.scatter(
                spots,
                gpr,
                color=GPR_COLOR,
                s=13,
                alpha=1.0,
                edgecolors="none",
                label="GPR",
                zorder=3,
            )
            if spots.min() <= strike <= spots.max():
                ax.axvline(
                    strike,
                    color="black",
                    linewidth=0.65,
                    linestyle="dashed",
                    alpha=0.7,
                )

            if row_index == 0:
                ax.set_title(rf"$T={maturity:g}$", fontsize=9)
            if row_index == len(GREEKS) - 1:
                ax.set_xlabel(r"$S_0$", fontsize=9)

    if show_row_labels:
        row_label_y = [0.79, 0.505, 0.22] if len(selected_maturities) == 1 else [0.803, 0.515, 0.226]
        for (_, label), y_position in zip(GREEKS, row_label_y, strict=True):
            fig.text(
                0.032,
                y_position,
                label,
                va="center",
                ha="center",
                fontsize=14,
            )

    if reference_handle is None or gpr_handle is None:
        raise ValueError("No plot handles were created.")
    fig.legend(
        [reference_handle, gpr_handle],
        ["Reference", "GPR"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.018),
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_single_maturity_figure5_4(
    grouped: dict[float, list[dict[str, float]]],
    output_path: Path,
    *,
    maturity: float,
    strike: float,
) -> None:
    """Create a Figure 5.4-style triangular layout for one maturity."""

    fig = plt.figure(figsize=(10.6, 7.5), dpi=FIGURE_DPI)
    axes = [
        fig.add_axes([0.305, 0.54, 0.39, 0.31]),
        fig.add_axes([0.08, 0.16, 0.39, 0.31]),
        fig.add_axes([0.52, 0.16, 0.39, 0.31]),
    ]

    rows = grouped[maturity]
    spots = np.array([row["S0"] for row in rows], dtype=float)
    reference_handle = None
    gpr_handle = None

    for ax, (greek, label) in zip(axes, GREEKS, strict=True):
        style_axis(ax)
        reference = np.array([row[f"{greek}_reference"] for row in rows], dtype=float)
        gpr = np.array([row[f"{greek}_gpr"] for row in rows], dtype=float)

        (reference_handle,) = ax.plot(
            spots,
            reference,
            color=REFERENCE_COLOR,
            linewidth=1.2,
            label="Reference",
            zorder=4,
        )
        gpr_handle = ax.scatter(
            spots,
            gpr,
            color=GPR_COLOR,
            s=18,
            alpha=1.0,
            edgecolors="none",
            linewidths=0.0,
            label="GPR",
            zorder=3,
        )
        if spots.min() <= strike <= spots.max():
            ax.axvline(
                strike,
                color="black",
                linewidth=0.65,
                linestyle="dashed",
                alpha=0.7,
            )

        ax.set_title(label, fontsize=12, pad=4)
        ax.set_xlabel(r"$S_0$", fontsize=8)

    if reference_handle is None or gpr_handle is None:
        raise ValueError("No plot handles were created.")
    fig.legend(
        [reference_handle, gpr_handle],
        ["Reference", "GPR"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.065),
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Plot Merton reference Greeks against GPR Greeks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/merton_jump_diffusion/greeks/predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graphs/numex/Figure7_5.pdf"),
    )
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--no-row-labels", action="store_true")
    parser.add_argument(
        "--maturities",
        type=float,
        nargs="+",
        default=list(DEFAULT_MATURITIES),
        help="Selected maturities to plot, e.g. --maturities 0.5",
    )
    return parser


def _figure_size(n_maturities: int) -> tuple[float, float]:
    if n_maturities == 1:
        return (4.8, 7.2)
    return (11.0, 8.0)


def main() -> None:
    """Load predictions and save the Greek comparison figure."""

    args = build_parser().parse_args()
    selected_maturities = tuple(args.maturities)
    grouped = load_prediction_rows(args.input, selected_maturities)
    plot_greeks(
        grouped,
        args.output,
        selected_maturities=selected_maturities,
        strike=args.strike,
        show_row_labels=not args.no_row_labels,
    )
    print(f"Saved {args.output}")

    if args.png:
        png_output = args.output.with_suffix(".png")
        plot_greeks(
            grouped,
            png_output,
            selected_maturities=selected_maturities,
            strike=args.strike,
            show_row_labels=not args.no_row_labels,
        )
        print(f"Saved {png_output}")


if __name__ == "__main__":
    main()
