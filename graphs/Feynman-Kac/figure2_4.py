import numpy as np
import matplotlib.pyplot as plt


def call_payoff(S, K):
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    return np.maximum(K - S, 0.0)


def style_axis(ax, x_min, x_max):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3)
    ax.grid(False)
    ax.set_xlim(x_min, x_max)



def main():
    K = 100.0
    S = np.linspace(0.0, 200.0, 1001)

    long_call = call_payoff(S, K)
    long_put = put_payoff(S, K)
    short_call = -long_call
    short_put = -long_put

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

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.ravel()

    payoffs = [long_call, long_put, short_call, short_put]
    titles = [
        "Long European call payoff",
        "Long European put payoff",
        "Short European call payoff",
        "Short European put payoff",
    ]

    for ax, y, title in zip(axes, payoffs, titles):
        ax.plot(S, y, color="k", linewidth=1.0)
        ax.axhline(0.0, color="k", linewidth=0.6, linestyle="dotted")
        ax.axvline(K, color="k", linewidth=0.6, linestyle="dotted")
        style_axis(ax, S.min(), S.max())
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Underlying price at maturity $S_T$", fontsize=8)
        ax.set_ylabel("Payoff", fontsize=8)

    plt.tight_layout()
    plt.savefig("graphs/figure2_4.pdf")
    plt.show()
    
""""
    # Individual files (4 separate graphs)
    individual_names = [
        "graphs/european_long_call_payoff.pdf",
        "graphs/european_long_put_payoff.pdf",
        "graphs/european_short_call_payoff.pdf",
        "graphs/european_short_put_payoff.pdf",
    ]

    for y, title, out_path in zip(payoffs, titles, individual_names):
        fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 3.5), dpi=150)
        ax_i.plot(S, y, color="k", linewidth=1.0)
        ax_i.axhline(0.0, color="k", linewidth=0.6, linestyle="dotted")
        ax_i.axvline(K, color="k", linewidth=0.6, linestyle="dotted")
        style_axis(ax_i, S.min(), S.max())
        ax_i.set_title(title, fontsize=8)
        ax_i.set_xlabel("Underlying price at maturity $S_T$", fontsize=8)
        ax_i.set_ylabel("Payoff", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig_i)

    plt.close(fig)
"""

if __name__ == "__main__":
    main()