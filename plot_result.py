import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = "output_plots"

'''
This is to plot loss and reward while training and plot mean waiting time, queue time, etc
BEFORE RUNNING THIS FILE, make sure that all model is in the system. INCLUDING PRESSURE REWARD SYSTEM
'''

PATHS = {
    "Local":                        "outputs/local_peak/results.json",
    "Cooperative":                  "outputs/cooperative_peak/results.json",
    "Fairness":                     "outputs/fairness_peak/results.json",
    "Local_OffPeak":                "outputs/local_off_peak/results.json",
    "Cooperative_OffPeak":          "outputs/cooperative_off_peak/results.json",
    "Fairness_OffPeak":             "outputs/fairness_off_peak/results.json",
    
    "Pressure_Local":               "outputs/pressure_local_peak/results.json",
    "Pressure_Cooperative":         "outputs/pressure_cooperative_peak/results.json",
    "Pressure_Fairness":            "outputs/pressure_fairness_peak/results.json",
    "Pressure_Local_OffPeak":       "outputs/pressure_local_off_peak/results.json",
    "Pressure_Cooperative_OffPeak": "outputs/pressure_cooperative_off_peak/results.json",
    "Pressure_Fairness_OffPeak":    "outputs/pressure_fairness_off_peak/results.json",
}

KPI_PATHS_PEAK = {}
KPI_PATHS_OFFPEAK = {}
'''
result would look like
{
    'Cooperative_Results.json': 'evaluations/cooperative_peak_results.json', 
    'Fairness_Results.json': 'evaluations/fairness_peak_results.json', 
    'Local_Results.json': 'evaluations/local_peak_results.json', 
    'Pressure_Cooperative_Results.json': 'evaluations/pressure_cooperative_peak_results.json', 
    'Pressure_Fairness_Results.json': 'evaluations/pressure_fairness_peak_results.json', 
    'Pressure_Local_Results.json': 'evaluations/pressure_local_peak_results.json'
}
'''
evaluation_dir = "evaluations/"
if os.path.isdir(evaluation_dir):
    for filename in os.listdir(evaluation_dir):
        if not filename.endswith(".json"):
            continue
        name = filename.replace("_result.json", "")
        if "_off_peak" in name:
            key = name.replace("_off_peak", "")
            # title-case it to match your COLORS keys
            key = "_".join(w.capitalize() for w in key.split("_"))
            KPI_PATHS_OFFPEAK[key] = evaluation_dir + filename
        elif "_peak" in name:
            key = name.replace("_peak", "")
            key = "_".join(w.capitalize() for w in key.split("_"))
            KPI_PATHS_PEAK[key] = evaluation_dir + filename

print(KPI_PATHS_PEAK)

COLORS = {
    "Local":       "#4C9BE8",
    "Cooperative": "#E8774C",
    "Fairness":    "#4CE89B",
    "Local_OffPeak":        "#1A5FA8",
    "Cooperative_OffPeak":  "#A83A10",
    "Fairness_OffPeak":     "#1A9B5F",

    "Pressure_Local":          "#9BC7F5",
    "Pressure_Cooperative":    "#F5B39B",
    "Pressure_Fairness":       "#9BF5C7",
    "Pressure_Local_OffPeak":       "#5F8FD6",
    "Pressure_Cooperative_OffPeak": "#D67A5F",
    "Pressure_Fairness_OffPeak":    "#5FD69B",
}

POLICY_COLORS = {
    "trained_dqn":    "#2563EB",   # bold blue  — highlighted
    "fixed_time":     "#94A3B8",   # slate grey
    "random_legal":   "#FCA5A5",   # soft red
    "greedy_queue":   "#6EE7B7",   # soft green
    "webster_static": "#FCD34D",   # soft amber
}
 
POLICY_LABELS = {
    "trained_dqn":    "Trained DQN",
    "fixed_time":     "Fixed Time",
    "random_legal":   "Random Legal",
    "greedy_queue":   "Greedy Queue",
    "webster_static": "Webster Static",
}

# Desired policy order in every bar cluster
POLICY_ORDER = ["trained_dqn", "fixed_time", "random_legal", "greedy_queue", "webster_static"]
 
# KPI metadata: key → (y-axis label, title suffix)
KPI_META = {
    "mean_waiting_time":  ("Mean Waiting Time (s)",    "Mean Waiting Time"),
    "mean_queue_length":  ("Mean Queue Length (veh)",  "Mean Queue Length"),
    "max_lane_wait":      ("Max Lane Wait (s)",         "Max Lane Wait"),
    "total_waiting_time": ("Total Waiting Time (s)",    "Total Waiting Time"),
}

DISPLAY_NAMES = {
    "Fairness_Results.json": "Fairness",
    "Local_Results.json": "Local",
    "Cooperative_Results.json": "Cooperative",
    "Pressure_Fairness_Results.json": "Pressure Fairness",
    "Pressure_Local_Results.json": "Pressure Local",
    "Pressure_Cooperative_Results.json": "Pressure Cooperative",
}

def _key_from_filename(filename: str, suffix: str) -> str:
    """
    Convert a filename like 'pressure_local_peak_result.json' into a COLORS-style
    key like 'Pressure_Local', given suffix='_peak' or '_off_peak'.
    """
    name = filename.replace("_result.json", "").replace(suffix, "")
    return "_".join(w.capitalize() for w in name.split("_"))

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
    ax.set_ylabel("Normalised Loss [0–1]", fontsize=11)
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


def plot_rewards(data: dict[str, dict], filename:str) -> None:
    labels = list(data.keys())
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
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    print("  Saved:", filename)
    plt.close(fig)
 
 
def plot_loss(data: dict[str, dict]) -> None:
    """All on one graph — loss is comparable across reward functions."""
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
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "loss.png"), dpi=150)
    print("  Saved: loss.png")
    plt.close(fig)

# ── KPI bar-chart helper ──────────────────────────────────────────────────────

def _get_dqn_mean(model_key, kpi_paths, kpi_key):
    try:
        kpi_data = load(kpi_paths[model_key])
        return kpi_data["policies"]["trained_dqn"]["aggregate"][kpi_key]["mean"]
    except (KeyError, FileNotFoundError):
        return float("inf")
    
def _plot_kpi(kpi_key: str, kpi_paths: dict[str, str], scenario_label: str, subfolder:str | None = None) -> None:
    """
    Draw a grouped bar chart for one KPI and one scenario (peak / off-peak).
 
    X groups  = reward-function models (keys of kpi_paths)
    Bars      = policies (POLICY_ORDER)
    Error bars = ± 1 std from aggregate stats
    DQN bar   = full opacity + black edge; baselines = 0.65 opacity, no edge
    """
    y_label, title_suffix = KPI_META[kpi_key]
 
    model_keys = sorted(kpi_paths.keys(), key=lambda k: _get_dqn_mean(k, kpi_paths, kpi_key))
    n_models   = len(model_keys)
    n_policies = len(POLICY_ORDER)
 

    bar_width = 0.14
    group_gap = 0.08
    group_width = n_policies * bar_width + group_gap
    x_centers   = np.arange(n_models) * group_width

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.2), 6))
    
    for p_idx, policy in enumerate(POLICY_ORDER):
        means, stds = [], []
        for model_key in model_keys:
            try:
                kpi_data = load(kpi_paths[model_key])
                agg      = kpi_data["policies"][policy]["aggregate"][kpi_key]
                means.append(agg["mean"])
                stds.append(agg["std"])
            except (KeyError, FileNotFoundError):
                means.append(0.0)
                stds.append(0.0)
 
        offsets = x_centers + (p_idx - n_policies / 2 + 0.5) * bar_width
        color    = POLICY_COLORS[policy]
        is_dqn   = policy == "trained_dqn"
        alpha    = 1.0  if is_dqn else 0.65
        edgecolor = "black" if is_dqn else "none"
        lw        = 1.2   if is_dqn else 0.0
        zorder    = 3     if is_dqn else 2
 
        ax.bar(
            offsets, means,
            width=bar_width,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=lw,
            zorder=zorder,
            label=POLICY_LABELS[policy],
            yerr=stds,
            capsize=3,
            error_kw=dict(elinewidth=1.0, ecolor="black", capthick=1.0, alpha=0.7),
        )

        for x, y, stds in zip(offsets, means, stds):
            ax.text(x, y + stds + max(means) * 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
 
    fixed_time_means = []
    for model_key in model_keys:
        try:
            kpi_data = load(kpi_paths[model_key])
            agg = kpi_data["policies"]["fixed_time"]["aggregate"][kpi_key]
            fixed_time_means.append(agg["mean"])
        except (KeyError, FileNotFoundError):
            fixed_time_means.append(0.0)

    avg_fixed = np.mean(fixed_time_means)
    ax.axhline(avg_fixed, color="#94A3B8", linewidth=1.5, 
            linestyle=":", alpha=0.8, label="Fixed Time avg (baseline)", zorder=1)
    

    ax.set_xticks(x_centers)
    ax.set_xticklabels( [DISPLAY_NAMES.get(k, k) for k in model_keys], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        f"{title_suffix} — {scenario_label}",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
 
    safe_kpi   = kpi_key.replace(" ", "_")
    safe_scen  = scenario_label.lower().replace(" ", "_").replace("-", "_")
    filename   = f"kpi_{safe_kpi}_{safe_scen}.png"
    output_dir = None
    if subfolder:
        output_dir = os.path.join(OUTPUT_DIR, subfolder)
    else:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)




 
# ── Public KPI plot functions ─────────────────────────────────────────────────
 
def plot_mean_wait_time() -> None:
    if KPI_PATHS_PEAK:
        _plot_kpi("mean_waiting_time", KPI_PATHS_PEAK,    "Peak")
    if KPI_PATHS_OFFPEAK:
        _plot_kpi("mean_waiting_time", KPI_PATHS_OFFPEAK, "Off-Peak")
 
 
def plot_mean_queue_time() -> None:
    if KPI_PATHS_PEAK:
        _plot_kpi("mean_queue_length", KPI_PATHS_PEAK,    "Peak")
    if KPI_PATHS_OFFPEAK:
        _plot_kpi("mean_queue_length", KPI_PATHS_OFFPEAK, "Off-Peak")
 
 
def plot_total_waiting_time() -> None:
    if KPI_PATHS_PEAK:
        _plot_kpi("total_waiting_time", KPI_PATHS_PEAK,    "Peak")
    if KPI_PATHS_OFFPEAK:
        _plot_kpi("total_waiting_time", KPI_PATHS_OFFPEAK, "Off-Peak")
 
 
def plot_max_waiting_time() -> None:
    if KPI_PATHS_PEAK:
        _plot_kpi("max_lane_wait", KPI_PATHS_PEAK,    "Peak")
    if KPI_PATHS_OFFPEAK:
        _plot_kpi("max_lane_wait", KPI_PATHS_OFFPEAK, "Off-Peak")

ORIGINAL_KEYS = {"Local_Results.json", "Cooperative_Results.json", "Fairness_Results.json"}
def plot_kpi_original() -> None:
    print('plotting kpi originals.')
    peak_paths    = {k: v for k, v in KPI_PATHS_PEAK.items() if k in ORIGINAL_KEYS}
    offpeak_paths = {k: v for k, v in KPI_PATHS_OFFPEAK.items() if k in ORIGINAL_KEYS}

    for kpi_key in KPI_META:
        if peak_paths:
            _plot_kpi(kpi_key, peak_paths, "Peak-Original-Reward", "original_rewards")
        if offpeak_paths:
            _plot_kpi(kpi_key, offpeak_paths, "Off-Peak-Original-Reward", "original_rewards")

PRESSURE_KEYS =  {"Pressure_Local_Results.json", "Pressure_Cooperative_Results.json", "Pressure_Fairness_Results.json"}
def plot_kpi_pressure() -> None:
    print('plotting kpi pressure')
    peak_paths      = {k : v for k, v in KPI_PATHS_PEAK.items() if k in PRESSURE_KEYS}
    offpeak_paths   = {k : v for k, v in KPI_PATHS_OFFPEAK.items() if k in PRESSURE_KEYS}

    for kpi_key in KPI_META:
        if peak_paths:
            _plot_kpi(kpi_key, peak_paths, "Peak-Pressure-Reward", "pressure_rewards")
        if offpeak_paths:
            _plot_kpi(kpi_key, offpeak_paths, "Off-Peak-Pressure-Reward", "pressure_rewards")
'''
    Total Plot Image
    wait time : peak and off peak 2 image
    queue amount : peak and off peak 2 image
    total wait time : peak and off peak 2 image
    max_wait_time : peak and off peak 2 image
    total is 8 image (kpi) + 5 image (reward + loss)
'''
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
    
    original_reward = {
        k: v for k, v in data.items()
        if not k.startswith('Pressure')
    }

    pressure_reward = {
        k: v for k, v in data.items()
        if k.startswith("Pressure")
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # plot the original reward system
    plot_rewards(original_reward, 'rewards_original.png')

    # plot the pressure reward system
    plot_rewards(pressure_reward, 'rewards_pressure.png')

    # plot all the loss
    plot_loss(data)
    plot_rewards_normalized(data)
    plot_loss_normalized(data)

    # After this section is is the KPIs.

    plot_mean_wait_time()
    plot_mean_queue_time()
    plot_total_waiting_time()
    plot_max_waiting_time()

    plot_kpi_original()
    plot_kpi_pressure()
    
    print("Done.")

if __name__ == '__main__':
    main()