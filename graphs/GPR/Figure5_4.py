from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rbf_covariance(x1, x2, signal_variance=1.0, lengthscale=0.22):
    distances = np.subtract.outer(x1, x2)
    return signal_variance * np.exp(-0.5 * (distances / lengthscale) ** 2)


def posterior_gp(
    x_train,
    y_train,
    x_test,
    signal_variance=1.0,
    lengthscale=0.22,
    noise_variance=1e-6,
):
    train_covariance = rbf_covariance(
        x_train,
        x_train,
        signal_variance=signal_variance,
        lengthscale=lengthscale,
    )
    cross_covariance = rbf_covariance(
        x_train,
        x_test,
        signal_variance=signal_variance,
        lengthscale=lengthscale,
    )
    test_covariance = rbf_covariance(
        x_test,
        x_test,
        signal_variance=signal_variance,
        lengthscale=lengthscale,
    )

    stabilized_train_covariance = train_covariance + (noise_variance + 1e-10) * np.eye(x_train.size)
    cholesky_factor = np.linalg.cholesky(stabilized_train_covariance)

    alpha = np.linalg.solve(
        cholesky_factor.T,
        np.linalg.solve(cholesky_factor, y_train),
    )
    posterior_mean = cross_covariance.T @ alpha

    solved_cross_covariance = np.linalg.solve(cholesky_factor, cross_covariance)
    posterior_covariance = test_covariance - solved_cross_covariance.T @ solved_cross_covariance
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)

    return posterior_mean, posterior_covariance


def sample_paths(mean, covariance, num_paths=3, seed=0):
    rng = np.random.default_rng(seed)
    stabilized_covariance = covariance + 1e-10 * np.eye(covariance.shape[0])
    cholesky_factor = np.linalg.cholesky(stabilized_covariance)
    standard_normals = rng.normal(size=(covariance.shape[0], num_paths))
    return (mean[:, None] + cholesky_factor @ standard_normals).T


def style_axis(ax, x_min, x_max, y_min, y_max):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def main():
    x = np.linspace(0.0, 1.0, 500)
    x_train = np.array([0.04, 0.10, 0.17, 0.26, 0.37, 0.49, 0.57, 0.72, 0.88, 0.95])
    y_train = np.array([1.45, 0.55, -1.05, -0.25, 1.35, -1.85, -0.55, 1.25, 0.45, -0.20])

    panel_specs = [
        {
            "title": r"$\ell = 0.07$, $\sigma_f = 1$",
            "lengthscale": 0.07,
            "signal_variance": 1.0,
            "seed": 2403164,
        },
        {
            "title": r"$\ell = 0.03$, $\sigma_f = 1.2$",
            "lengthscale": 0.03,
            "signal_variance": 1.2,
            "seed": 2403165,
        },
        {
            "title": r"$\ell = 0.11$, $\sigma_f = 1.9$",
            "lengthscale": 0.11,
            "signal_variance": 1.9,
            "seed": 2403166,
        },
    ]

    panels = []
    for spec in panel_specs:
        posterior_mean, posterior_covariance = posterior_gp(
            x_train,
            y_train,
            x,
            signal_variance=spec["signal_variance"],
            lengthscale=spec["lengthscale"],
            noise_variance=1e-6,
        )
        posterior_std = np.sqrt(np.clip(np.diag(posterior_covariance), 0.0, None))
        posterior_paths = sample_paths(
            posterior_mean,
            posterior_covariance,
            num_paths=3,
            seed=spec["seed"],
        )
        panels.append(
            {
                "title": spec["title"],
                "mean": posterior_mean,
                "std": posterior_std,
                "paths": posterior_paths,
            }
        )

    y_extent = max(
        max(np.max(np.abs(panel["paths"])) for panel in panels),
        max(np.max(np.abs(panel["mean"] + 2.0 * panel["std"])) for panel in panels),
        max(np.max(np.abs(panel["mean"] - 2.0 * panel["std"])) for panel in panels),
        np.max(np.abs(y_train)),
    ) + 0.25
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

    fig = plt.figure(figsize=(10.6, 7.5), dpi=150)
    axes = [
        fig.add_axes([0.305, 0.54, 0.39, 0.31]),
        fig.add_axes([0.08, 0.16, 0.39, 0.31]),
        fig.add_axes([0.52, 0.16, 0.39, 0.31]),
    ]

    band_color = "#d9e1df"
    mean_color = "#274c5e"
    path_colors = ["#7f97a6", "#a69282", "#8ea58a"]

    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        style_axis(ax, x.min(), x.max(), y_min, y_max)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.fill_between(
            x,
            panel["mean"] - 2.0 * panel["std"],
            panel["mean"] + 2.0 * panel["std"],
            color=band_color,
            alpha=0.8,
            linewidth=0.0,
            zorder=0,
        )
        ax.plot(x, panel["mean"], color=mean_color, linewidth=1.2, linestyle="dashed")
        for path, color in zip(panel["paths"], path_colors):
            ax.plot(x, path, color=color, linewidth=1.05, alpha=1.0)
        ax.scatter(
            x_train,
            y_train,
            color=mean_color,
            s=22,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.set_title(panel["title"], fontsize=8)
        ax.set_xlabel(r"$x$", fontsize=8)
        ax.set_ylabel(r"$f(x)$", fontsize=8, labelpad=2)
        if panel_index == 2:
            ax.yaxis.set_label_coords(-0.06, 0.5)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
