import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

local_peak = "outputs/local_peak_20260602_142746/results.json"
cooperative_peak = "outputs/cooperative_peak_20260602_170004/results.json" 
fairness_peak = "outputs/fairness_peak_20260604_134036/results.json"

PATHS = {
    "Local":       local_peak,
    "Cooperative": cooperative_peak,
    "Fairness":    fairness_peak,
}
 
COLORS = {
    "Local":       "#4C9BE8",
    "Cooperative": "#E8774C",
    "Fairness":    "#4CE89B",
}
 
SMOOTH_WINDOW = 20 

def load(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
 
 
def smooth(values: list, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    # 'valid' shortens the array; pad with NaN so indices align
    pad = window - 1
    smoothed = np.convolve(arr, kernel, mode="valid")
    return np.concatenate([np.full(pad, np.nan), smoothed])


def plot_rewards(data: dict[str, dict]) -> None:
    labels = list(data.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)

    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        result = data[label]
        rewards = result["episode_rewards"]
        episodes = np.arange(1, len(rewards) + 1)
        color = COLORS[label]

        ax.plot(episodes, rewards, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(episodes, smooth(rewards, SMOOTH_WINDOW),
                color=color, linewidth=2)

        ax.set_title(f"{label} Reward", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Avg Reward", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle("Training Reward per Reward Function\n(not comparable across subplots — different equations)",
                    fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("output_plots/rewards.png", dpi=150, bbox_inches="tight")
    print("  Saved: rewards.png")
    plt.close(fig)
 
 
def plot_loss(data: dict[str, dict]) -> None:
    """All on one graph — MSE loss is comparable across reward functions."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for label, result in data.items():
        losses = result["episode_losses"]
        episodes = np.arange(1, len(losses) + 1)
        color = COLORS[label]

        nonzero = [(e, l) for e, l in zip(episodes, losses) if l > 0]
        if not nonzero:
            continue
        ep_nz, l_nz = zip(*nonzero)
        ep_nz, l_nz = np.array(ep_nz), list(l_nz)

        ax.plot(ep_nz, l_nz, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(ep_nz, smooth(l_nz, SMOOTH_WINDOW),
                color=color, linewidth=2, label=label)

    ax.set_title("Training Loss — All Reward Functions", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("output_plots/loss.png", dpi=150)
    print("  Saved: loss.png")
    plt.close(fig)


def main():
    data = {}
    for label, path in PATHS.items():
        try:
            data[label] = load(path)
            episodes = len(data[label]["episode_rewards"])
            print(f"  Loaded {label:12s} — {episodes} episodes  ({path})")
        except FileNotFoundError:
            print(f"  SKIP {label}: file not found — {path}")
 
    if not data:
        print("No data loaded, exiting.")
        return
 
    plot_rewards(data)
    plot_loss(data)
    print("Done.")

if __name__ == '__main__':
    main()