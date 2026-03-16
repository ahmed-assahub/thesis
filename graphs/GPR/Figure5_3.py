from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

PATH_COLORS = ["#274c5e", "#7f97a6", "#a69282", "#8ea58a", "#b7a395"]


def rbf_covariance(x1, x2, variance=1.0, lengthscale=0.22):
    distances = np.subtract.outer(x1, x2)
    return variance * np.exp(-0.5 * (distances / lengthscale) ** 2)


def matern_five_halves_covariance(x1, x2, variance=1.0, lengthscale=0.22):
    distances = np.abs(np.subtract.outer(x1, x2))
    scaled_distances = np.sqrt(5.0) * distances / lengthscale
    polynomial = 1.0 + scaled_distances + (scaled_distances**2) / 3.0
    return variance * polynomial * np.exp(-scaled_distances)


def matern_one_half_covariance(x1, x2, variance=1.0, lengthscale=0.22):
    distances = np.abs(np.subtract.outer(x1, x2))
    return variance * np.exp(-distances / lengthscale)


def sample_paths(covariance, num_paths=5, seed=0):
    rng = np.random.default_rng(seed)
    stabilized_covariance = covariance + 1e-10 * np.eye(covariance.shape[0])
    cholesky_factor = np.linalg.cholesky(stabilized_covariance)
    standard_normals = rng.normal(size=(covariance.shape[0], num_paths))
    return (cholesky_factor @ standard_normals).T


def style_axis(ax, x_min, x_max, y_min, y_max):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0.0, color="0.65", linewidth=0.6, linestyle="dotted", zorder=0)


def correlation_at_distance(covariance_function, distance, lengthscale):
    return covariance_function(
        np.array([0.0]),
        np.array([distance]),
        variance=1.0,
        lengthscale=lengthscale,
    )[0, 0]


def matched_lengthscale(covariance_function, reference_distance, reference_correlation):
    low = 1e-3
    high = 1.0

    while correlation_at_distance(covariance_function, reference_distance, high) < reference_correlation:
        high *= 2.0

    for _ in range(60):
        midpoint = 0.5 * (low + high)
        if correlation_at_distance(covariance_function, reference_distance, midpoint) < reference_correlation:
            low = midpoint
        else:
            high = midpoint

    return 0.5 * (low + high)


def build_kernel_panels(
    x,
    variance=1.0,
    num_paths=5,
    reference_distance=0.25,
    reference_correlation=0.4,
):
    kernel_specs = [
        ("RBF kernel", rbf_covariance, 2403161),
        (r"Matern kernel ($\nu = 5/2$)", matern_five_halves_covariance, 2403162),
        (r"Matern kernel ($\nu = 1/2$)", matern_one_half_covariance, 2403163),
    ]

    panels = []
    for title, covariance_function, seed in kernel_specs:
        lengthscale = matched_lengthscale(
            covariance_function,
            reference_distance=reference_distance,
            reference_correlation=reference_correlation,
        )
        covariance = covariance_function(
            x,
            x,
            variance=variance,
            lengthscale=lengthscale,
        )
        panels.append(
            {
                "title": title,
                "paths": sample_paths(covariance, num_paths=num_paths, seed=seed),
            }
        )

    return panels


def main():
    x = np.linspace(0.0, 1.0, 320)

    # Match kernels by the same correlation at a common reference distance.
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, sharey=True)

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
