from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Figure5_3 import PATH_COLORS, PLOT_STYLE, build_kernel_panels, style_axis


def main():
    x = np.linspace(0.0, 1.0, 320)

    panels = build_kernel_panels(
        x,
        variance=1.0,
        num_paths=5,
        reference_distance=0.25,
        reference_correlation=0.4,
    )

    y_extent = max(np.max(np.abs(panel["paths"])) for panel in panels) + 0.25
    y_min = -y_extent
    y_max = y_extent

    output_path = Path(__file__).with_suffix(".pdf")

    plt.rcParams.update(PLOT_STYLE)

    fig = plt.figure(figsize=(10, 6.2), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(grid[1, :], sharey=ax_left)
    axes = [ax_left, ax_right, ax_bottom]

    for axis, panel in zip(axes, panels):
        style_axis(axis, x.min(), x.max(), y_min, y_max)
        for path, color in zip(panel["paths"], PATH_COLORS):
            axis.plot(x, path, color=color, linewidth=1.0, alpha=1.0)
        axis.set_title(panel["title"], fontsize=8)
        axis.set_xlabel(r"$x$", fontsize=8)
        axis.set_ylabel(r"$f(x)$", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
