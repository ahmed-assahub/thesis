import numpy as np
import matplotlib.pyplot as plt


def poisson_jump_times(T=10.0, lam=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = 0.0
    times = []
    while True:
        t += rng.exponential(scale=1.0 / lam)
        if t > T:
            break
        times.append(t)
    return np.array(times)


def normal_jump_sizes(n_jumps, mu=1.0, sigma=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n_jumps)


def compensated_levels(jump_times, jump_sizes, lam=1.0, mu=1.0):
    # X_t = sum_{i<=N_t} Y_i - lambda * t * E[Y_1], with E[Y_1] = mu.
    return np.cumsum(jump_sizes) - lam * mu * jump_times


def plot_compensated_jump_process(ax, jump_times, levels_after, T, lam=1.0, mu=1.0, lw_h=1.0, lw_v=0.6):
    t_prev = 0.0
    y_prev = 0.0
    drift = -lam * mu

    for tj, y_after in zip(jump_times, levels_after):
        y_before = y_prev + drift * (tj - t_prev)
        ax.plot([t_prev, tj], [y_prev, y_before], color="k", linewidth=lw_h)
        ax.vlines(tj, y_before, y_after, colors="k", linestyles="dotted", linewidth=lw_v)
        t_prev = tj
        y_prev = y_after

    y_T = y_prev + drift * (T - t_prev)
    ax.plot([t_prev, T], [y_prev, y_T], color="k", linewidth=lw_h)


def main():
    T = 10.0
    lam_left = 1.0
    lam_right = 50
    mu = 1.0
    sigma = 1.5

    seed_times_left = 210234384
    seed_sizes_left = 134982734
    seed_times_right = 210234385
    seed_sizes_right = 13498735

    jump_times_left = poisson_jump_times(T=T, lam=lam_left, seed=seed_times_left)
    jump_sizes_left = normal_jump_sizes(len(jump_times_left), mu=mu, sigma=sigma, seed=seed_sizes_left)
    levels_left = compensated_levels(jump_times_left, jump_sizes_left, lam=lam_left, mu=mu)

    jump_times_right = poisson_jump_times(T=T, lam=lam_right, seed=seed_times_right)
    jump_sizes_right = normal_jump_sizes(len(jump_times_right), mu=mu, sigma=sigma, seed=seed_sizes_right)
    levels_right = compensated_levels(jump_times_right, jump_sizes_right, lam=lam_right, mu=mu)
    e_y2 = mu**2 + sigma**2
    right_scale = np.sqrt(lam_right * e_y2)
    levels_right_rescaled = levels_right / right_scale

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
        ax.grid(False)
        ax.set_xlim(0, T)

    plot_compensated_jump_process(axes[0], jump_times_left, levels_left, T, lam=lam_left, mu=mu, lw_h=1.0, lw_v=0.6)
    axes[0].set_title("Sample path of a compensated compound Poisson process", fontsize=8)

    plot_compensated_jump_process(
        axes[1],
        jump_times_right,
        levels_right_rescaled,
        T,
        lam=lam_right,
        mu=mu / right_scale,
        lw_h=1.0,
        lw_v=0.6,
    )
    axes[1].set_title("Rescaled compensated compound Poisson process", fontsize=8)

    plt.tight_layout()
    plt.savefig("graphs/figure2_3.pdf")
    plt.show()


if __name__ == "__main__":
    main()
