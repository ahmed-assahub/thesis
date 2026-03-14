from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rbf_covariance(x1, x2, variance=1.0, lengthscale=0.22):
    distances = np.subtract.outer(x1, x2)
    return variance * np.exp(-0.5 * (distances / lengthscale) ** 2)


def sample_paths(mean, covariance, num_paths=5, seed=0):
    rng = np.random.default_rng(seed)
    stabilized_covariance = covariance + 1e-10 * np.eye(covariance.shape[0])
    cholesky_factor = np.linalg.cholesky(stabilized_covariance)
    standard_normals = rng.normal(size=(covariance.shape[0], num_paths))
    return (mean[:, None] + cholesky_factor @ standard_normals).T


def posterior_gp(x_train, y_train, x_test, variance=1.0, lengthscale=0.22, noise_variance=1e-6):
    train_covariance = rbf_covariance(
        x_train,
        x_train,
        variance=variance,
        lengthscale=lengthscale,
    )
    cross_covariance = rbf_covariance(
        x_train,
        x_test,
        variance=variance,
        lengthscale=lengthscale,
    )
    test_covariance = rbf_covariance(
        x_test,
        x_test,
        variance=variance,
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


def style_axis(ax, x_min, x_max, y_min, y_max):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def main():
    x = np.linspace(0.0, 1.0, 320)
    variance = 1.0
    lengthscale = 0.22

    prior_mean = np.zeros_like(x)
    prior_covariance = rbf_covariance(
        x,
        x,
        variance=variance,
        lengthscale=lengthscale,
    )
    prior_std = np.sqrt(np.clip(np.diag(prior_covariance), 0.0, None))
    prior_paths = sample_paths(prior_mean, prior_covariance, num_paths=3, seed=2403151)

    x_train = np.array([0.16, 0.51, 0.84])
    y_train = np.array([0.95, -0.65, 0.75])
    posterior_mean, posterior_covariance = posterior_gp(
        x_train,
        y_train,
        x,
        variance=variance,
        lengthscale=lengthscale,
        noise_variance=1e-6,
    )
    posterior_std = np.sqrt(np.clip(np.diag(posterior_covariance), 0.0, None))
    posterior_paths = sample_paths(
        posterior_mean,
        posterior_covariance,
        num_paths=3,
        seed=2403152,
    )

    y_extent = max(
        np.max(np.abs(prior_paths)),
        np.max(np.abs(posterior_paths)),
        np.max(np.abs(prior_mean + 2.0 * prior_std)),
        np.max(np.abs(posterior_mean + 2.0 * posterior_std)),
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)

    style_axis(axes[0], x.min(), x.max(), y_min, y_max)
    style_axis(axes[1], x.min(), x.max(), y_min, y_max)

    prior_band_color = "#dbe3e6"
    posterior_band_color = "#d9e1df"
    mean_color = "#274c5e"
    path_colors = ["#7f97a6", "#a69282", "#8ea58a"]

    axes[0].fill_between(
        x,
        prior_mean - 2.0 * prior_std,
        prior_mean + 2.0 * prior_std,
        color=prior_band_color,
        alpha=0.8,
        linewidth=0.0,
        zorder=0,
    )
    axes[0].plot(x, prior_mean, color=mean_color, linewidth=1.1, linestyle="dashed")
    for path, color in zip(prior_paths, path_colors):
        axes[0].plot(x, path, color=color, linewidth=1.05, alpha=1.0)

    axes[0].set_title("prior", fontsize=8)
    axes[0].set_xlabel(r"$x$", fontsize=8)
    axes[0].set_ylabel(r"$f(x)$", fontsize=8)

    axes[1].fill_between(
        x,
        posterior_mean - 2.0 * posterior_std,
        posterior_mean + 2.0 * posterior_std,
        color=posterior_band_color,
        alpha=0.8,
        linewidth=0.0,
        zorder=0,
    )
    axes[1].plot(x, posterior_mean, color=mean_color, linewidth=1.2, linestyle="dashed")
    for path, color in zip(posterior_paths, path_colors):
        axes[1].plot(x, path, color=color, linewidth=1.05, alpha=1.0)

    axes[1].scatter(
        x_train,
        y_train,
        color=mean_color,
        s=22,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    axes[1].set_title("posterior", fontsize=8)
    axes[1].set_xlabel(r"$x$", fontsize=8)
    axes[1].set_ylabel(r"$f(x)$", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
