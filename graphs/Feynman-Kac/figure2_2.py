import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def poisson_jump_times(T=10.0, lam=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = 0.0
    times = []
    while True:
        t += rng.exponential(scale=1.0 / lam)
        if t > T:
            break
        times.append(t)
    return np.array(times)

def compound_poisson_jumps(n_jumps, mu=0.0, sigma=1.0, seed=1):
    """Jump sizes Y_i ~ N(mu, sigma^2) for a compound Poisson process."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n_jumps)

def plot_jump_process(ax, jump_times, levels, T, lw_h=0.8, lw_v=0.6):
    """
    Plot a càdlàg jump process given:
      - jump_times: times t_i
      - levels: process values just AFTER each jump (same length as jump_times),
               with level at time 0 assumed 0.
    Style:
      - horizontal segments solid
      - vertical jump markers dotted
    """
    t_prev = 0.0
    y_prev = 0.0

    for tj, y_after in zip(jump_times, levels):
        # horizontal segment up to the jump
        ax.hlines(y_prev, t_prev, tj, colors="k", linewidth=lw_h)
        # dotted vertical marker indicating the jump
        ax.vlines(tj, y_prev, y_after, colors="k", linestyles="dotted", linewidth=lw_v)

        t_prev = tj
        y_prev = y_after

    # final horizontal segment to T
    ax.hlines(y_prev, t_prev, T, colors="k", linewidth=lw_h)

def main():
    T = 10.0
    lam = 1.0

    mu = 0.0
    sigma = 2.0

    seed_times = 210234384
    seed_sizes = 134982734

    jump_times = poisson_jump_times(T=T, lam=lam, seed=seed_times)
    N = len(jump_times)

    levels_poisson = np.arange(1, N + 1, dtype=float)

    jump_sizes = compound_poisson_jumps(N, mu=mu, sigma=sigma, seed=seed_sizes)
    levels_compound = np.cumsum(jump_sizes)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(axis='both', labelsize=9, width=0.6, length=3)
        ax.grid(False)

    plot_jump_process(axes[0], jump_times, levels_poisson, T, lw_h=1.0, lw_v=0.6)
    axes[0].set_xlim(0, T)
    axes[0].set_title("Sample path of a Poisson process", fontsize=8)


    axes[0].set_ylim(0, max(1, N + 1))
    axes[0].yaxis.set_major_locator(MultipleLocator(1))


    plot_jump_process(axes[1], jump_times, levels_compound, T, lw_h=1.0, lw_v=0.6)
    axes[1].set_xlim(0, T)
    axes[1].set_title("Sample path of a compound Poisson process",fontsize=8)
    axes[1].yaxis.set_major_locator(MultipleLocator(1))


    plt.tight_layout()
    plt.savefig("graphs/figure2_2.pdf")
    plt.show()

if __name__ == "__main__":
    main()