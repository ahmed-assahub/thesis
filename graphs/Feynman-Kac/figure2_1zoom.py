import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=1.0, N=2000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, N + 1)
    dt = T / N
    dW = rng.normal(0.0, np.sqrt(dt), size=N)
    W = np.concatenate(([0.0], np.cumsum(dW)))
    return t, W

def main():
    T_bm = 10
    N_bm = 10000

    T_zoom = 2.0

    t, W = brownian_motion(T=T_bm, N=N_bm, seed=102384)

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
            ax.tick_params(axis='both', labelsize=9)
            ax.tick_params(axis='both', labelsize=9, width=0.6, length=3)

    axes[0].plot(t, W, color="k", linewidth=0.5)
    axes[0].set_xlim(0, 8) 
    axes[0].set_ylim(-2,3)
    axes[0].grid(False)
    axes[0].set_title("Sample path of a Wiener process", fontsize = 8)

    axes[1].plot(t,W, color = "k", linewidth=0.5)
    axes[1].set_xlim(0,T_zoom)
    axes[1].set_ylim(-7/6,1.75)
    axes[1].grid(False)
    axes[1].set_title("Zoomed sample path of a Wiener process", fontsize= 8)

    plt.tight_layout()
    plt.savefig("graphs/Figure2_1zoom.pdf")
    plt.show()
    
if __name__ == "__main__":
    main()