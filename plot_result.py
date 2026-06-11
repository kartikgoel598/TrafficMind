import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = "output_plots"

# WRITE THE FOLDER HERE
local_peak              = "outputs/local_peak_final/results.json"
cooperative_peak        = "outputs/cooperative_peak_final/results.json" 
fairness_peak           = "outputs/fairness_peak_final/results.json"
local_off_peak          = "outputs/local_off_peak_final/results.json"
cooperative_off_peak    = "outputs/cooperative_off_peak_final/results.json"
fairness_off_peak       = "outputs/fairness_off_peak_final/results.json"

PATHS = {
    "Local":                local_peak,
    "Cooperative":          cooperative_peak,
    "Fairness":             fairness_peak,
    "Local_OffPeak":        local_off_peak,
    "Cooperative_OffPeak":  cooperative_off_peak,
    "Fairness_OffPeak":     fairness_off_peak,
}
 
COLORS = {
    "Local":       "#4C9BE8",
    "Cooperative": "#E8774C",
    "Fairness":    "#4CE89B",
    "Local_OffPeak":        "#1A5FA8",
    "Cooperative_OffPeak":  "#A83A10",
    "Fairness_OffPeak":     "#1A9B5F",
}


def normalize(values):
    arr = np.array(values, dtype=float)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)
 
def plot_rewards_normalized(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, result in data.items():
        rewards = result["episode_rewards"]
        episodes = np.arange(1, len(rewards) + 1)
        normed = normalize(rewards)
        color = COLORS[label]
        ls = "--" if "OffPeak" in label else "-"
        ax.plot(episodes, normed, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(episodes, smooth(normed.tolist(), SMOOTH_WINDOW),
                color=color, linewidth=2, linestyle=ls, label=label)
    ax.set_title("Normalised Reward — All Conditions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Normalised Reward [0–1]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rewards_normalized.png"), dpi=150)
    print("  Saved: rewards_normalized.png")
    plt.close(fig)


def plot_loss_normalized(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, result in data.items():
        losses = result["episode_losses"]
        episodes = np.arange(1, len(losses) + 1)
        nonzero = [(e, l) for e, l in zip(episodes, losses) if l > 0]
        if not nonzero:
            continue
        ep_nz, l_nz = zip(*nonzero)
        normed = normalize(list(l_nz))
        color = COLORS[label]
        ls = "--" if "OffPeak" in label else "-"
        ax.plot(np.array(ep_nz), normed, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(np.array(ep_nz), smooth(normed.tolist(), SMOOTH_WINDOW),
                color=color, linewidth=2, linestyle=ls, label=label)
    ax.set_title("Normalised Loss — All Conditions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Normalised MSE Loss [0–1]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_normalized.png"), dpi=150)
    print("  Saved: loss_normalized.png")
    plt.close(fig)



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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes_flat = axes.flatten()

    for ax, label in zip(axes_flat, labels):
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
    fig.savefig(os.path.join(OUTPUT_DIR, "rewards.png"), dpi=150, bbox_inches="tight")
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
    fig.savefig(os.path.join(OUTPUT_DIR, "loss.png"), dpi=150)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_rewards(data)
    plot_loss(data)
    plot_rewards_normalized(data)
    plot_loss_normalized(data)
    print("Done.")

if __name__ == '__main__':
    main()