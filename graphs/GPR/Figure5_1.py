from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rbf_covariance(x, variance=1.0, lengthscale=0.18):
    distances = np.subtract.outer(x, x)
    return variance * np.exp(-0.5 * (distances / lengthscale) ** 2)


def sample_gp_paths(x, num_paths=6, variance=1.0, lengthscale=0.18, seed=0):
    covariance = rbf_covariance(x, variance=variance, lengthscale=lengthscale)
    covariance += 1e-10 * np.eye(x.size)
    rng = np.random.default_rng(seed)
    standard_normals = rng.normal(size=(x.size, num_paths))
    cholesky_factor = np.linalg.cholesky(covariance)
    return (cholesky_factor @ standard_normals).T


def normal_pdf(y, mean=0.0, variance=1.0):
    standard_deviation = np.sqrt(variance)
    z = (y - mean) / standard_deviation
    return np.exp(-0.5 * z**2) / (standard_deviation * np.sqrt(2.0 * np.pi))


def style_axis(ax, x_min, x_max, y_min, y_max):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0.0, color="0.65", linewidth=0.6, linestyle="dotted", zorder=0)


def main():
    x = np.linspace(0.0, 1.0, 320)
    variance = 1.0
    lengthscale = 0.22
    num_paths = 7
    seed = 240314535353

    paths = sample_gp_paths(
        x,
        num_paths=num_paths,
        variance=variance,
        lengthscale=lengthscale,
        seed=seed,
    )

    highlighted_index = 3
    highlighted_path = paths[highlighted_index]
    x_star_index = 165
    x_star = x[x_star_index]
    y_star = highlighted_path[x_star_index]

    y_extent = max(np.max(np.abs(paths)), 3.0 * np.sqrt(variance), abs(y_star)) + 0.25
    y_min = -y_extent
    y_max = y_extent

    output_path = Path(__file__).with_suffix(".pdf")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)

    style_axis(axes[0], x.min(), x.max(), y_min, y_max)
    style_axis(axes[1], x.min(), x.max(), y_min, y_max)

    path_colors = [
        "#c7d0d8",
        "#d6cec2",
        "#c8d4c6",
        "#274c5e",
        "#d8c8bb",
        "#bcc7cd",
        "#c9cec2",
    ]

    for path_index, (path, color) in enumerate(zip(paths, path_colors)):
        linewidth = 1.35 if path_index == highlighted_index else 0.9
        alpha = 1.0 if path_index == highlighted_index else 0.9
        axes[0].plot(x, path, color=color, linewidth=linewidth, alpha=alpha)

    axes[0].set_title("Sample paths of a Gaussian process", fontsize=8)
    axes[0].set_xlabel(r"$x$", fontsize=8)
    axes[0].set_ylabel(r"$f(x)$", fontsize=8)

    axes[1].plot(x, highlighted_path, color="#274c5e", linewidth=1.35)
    axes[1].axvline(x_star, color="k", linewidth=0.6, linestyle="dotted")
    axes[1].scatter([x_star], [y_star], color="#274c5e", s=18, zorder=4)

    y_density = np.linspace(y_min, y_max, 500)
    pdf = normal_pdf(y_density, mean=0.0, variance=variance)
    pdf /= pdf.max()

    density_base = x_star
    density_width = 0.2
    density_curve = density_base + density_width * pdf

    axes[1].fill_betweenx(
        y_density,
        density_base,
        density_curve,
        color="#d9e1df",
        alpha=0.85,
        linewidth=0.0,
    )
    axes[1].plot(density_curve, y_density, color="#6a7f88", linewidth=0.9)
    axes[1].set_title("A fixed input induces a Gaussian distribution", fontsize=8)
    axes[1].set_xlabel(r"$x$", fontsize=8)
    axes[1].set_ylabel(r"$f(x)$", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
