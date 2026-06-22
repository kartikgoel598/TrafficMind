import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traci

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent

from dotenv import load_dotenv
import sys 

load_dotenv()
sumo_home = os.getenv("Sumo_Home")
if not sumo_home:
    raise EnvironmentError("Sumo_Home not set in .env")
sys.path.append(os.path.join(sumo_home, "tools"))

INTERSECTIONS = ["J1", "J2", "J4", "J5"]
PROJECT_ROOT = os.path.dirname((os.path.abspath(__file__)))
HEATMAP_DIR = 'output_plots/heatmap'
STARVATION_DIR = 'output_plots/starvation'
DEFAULT_SEED = [42, 123, 456, 789, 1000]


MODELS = {
    # ── Peak ──────────────────────────────────────────────────────────────────
    "Local Peak":                    ("outputs/local_peak",                 "peak"),
    "Cooperative Peak":              ("outputs/cooperative_peak",            "peak"),
    "Fairness Peak":                 ("outputs/fairness_peak",               "peak"),
    "Pressure Local Peak":           ("outputs/pressure_local_peak",         "peak"),
    "Pressure Cooperative Peak":     ("outputs/pressure_cooperative_peak",   "peak"),
    "Pressure Fairness Peak":        ("outputs/pressure_fairness_peak",      "peak"),
    # ── Off-Peak ──────────────────────────────────────────────────────────────
    "Local Off-Peak":                ("outputs/local_off_peak",              "off_peak"),
    "Cooperative Off-Peak":          ("outputs/cooperative_off_peak",        "off_peak"),
    "Fairness Off-Peak":             ("outputs/fairness_off_peak",           "off_peak"),
    "Pressure Local Off-Peak":       ("outputs/pressure_local_off_peak",     "off_peak"),
    "Pressure Cooperative Off-Peak": ("outputs/pressure_cooperative_off_peak","off_peak"),
    "Pressure Fairness Off-Peak":    ("outputs/pressure_fairness_off_peak",  "off_peak"),
}

FAIRNESS_MODELS = {
    "Fairness Peak":             ("outputs/fairness_peak", "peak"),
    "Fairness Off-Peak":         ("outputs/fairness_off_peak", "off_peak"),
    "Pressure Fairness Peak":    ("outputs/pressure_fairness_peak", "peak"),
    "Pressure Fairness Off-Peak":("outputs/pressure_fairness_off_peak", "off_peak"), 
}

MODEL_COLORS = {
    "Fairness Peak":             "#E84C4C",
    "Fairness Off-Peak":         "#4C9BE8",
    "Pressure Fairness Peak":    "#E8C34C",
    "Pressure Fairness Off-Peak":"#9B4CE8", 
}

def load(path):
    with open(path,'r') as f:
        return json.load(f)
    
def get_env(scenario, reward_fn):
    config= 'peak.sumocfg' if scenario == "peak" else "off_peak.sumocfg"
    return SumoEnvironment(
        config_path = os.path.join(PROJECT_ROOT, 'sumo', 'configs', config),
        reward_fn = reward_fn,
        use_gui = False,
    )

def load_agents(env, models_dir):
    agents = {}
    for j in INTERSECTIONS:
        path = os.path.join(PROJECT_ROOT, models_dir, f"agent_{j}_final.pth")

        agent = DQNAgent(
            state_size=env.state_size, action_size=env.action_size, epsilon=0.0
        )
        agent.load(path)
        agents[j] = agent

    return agents

def select_actions(states, agents):
    return {
        j: agents[j].select_action(states[j])
        for j in INTERSECTIONS
    }

def run_episode_extended(env, agents, seed=0):
    env.seed = seed
    states = env.reset()

    wait_history = {j: [] for j in INTERSECTIONS}
    starvation_history = {j : [] for j in INTERSECTIONS}

    done = False

    while not done:
        actions = select_actions(states, agents)
        states, rewards, done, executed = env.step(actions)

        # heatmap and starvation data
        for j in INTERSECTIONS:
            total_wait = 0
            for lane in env.lanes[j]:
                total_wait += traci.lane.getWaitingTime(lane)

            wait_history[j].append(total_wait)
            starvation_history[j].append(max(env._red_time[j]))

    return wait_history, starvation_history


def collect_heatmap_data():
    peak_data, offpeak_data = {}, {}

    for label, (models_dir, scenario) in MODELS.items():
        print(f'running {label}')
        env = get_env(scenario, reward_fn="local")
        agents = load_agents(env, models_dir)

        all_wait_histories = []
        for ep in range(5):
            print(f'episode {ep+1}/5')
            env.seed = DEFAULT_SEED[ep]
            wait_history, _ = run_episode_extended(env, agents)
            all_wait_histories.append(wait_history)

        env.close()

        means = {
            j: float(np.mean([np.mean(ep[j]) for ep in all_wait_histories]))
            for j in INTERSECTIONS
        }
        if scenario == "peak":
            peak_data[label] = means
        else:
            offpeak_data[label] = means

    return peak_data, offpeak_data

def plot_heatmap_combined(data: dict, title: str, filename: str) -> None:
    """
    Rows = conditions (6), Cols = junctions (4)
    Cell = mean waiting time (s)
    """
    row_labels = list(data.keys())
    matrix = np.array([[data[label][j] for j in INTERSECTIONS] for label in row_labels])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=INTERSECTIONS,
        yticklabels=row_labels,
        cbar_kws={"label": "Mean Waiting Time (s)"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Intersection", fontsize=11)
    ax.set_ylabel("Condition", fontsize=11)
    fig.tight_layout()

    os.makedirs(HEATMAP_DIR, exist_ok=True)
    fig.savefig(os.path.join(HEATMAP_DIR, filename), dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


def run_heatmap_analysis():
    peak_data, offpeak_data = collect_heatmap_data()
    plot_heatmap_combined(peak_data, "Mean Waiting Time per Intersection Peak", "heatmap_peak.png")
    plot_heatmap_combined(offpeak_data, "Mean Waiting Time per Intersection Off-Peak", "heatmap_off_peak.png")

def collect_starvation_data():
    '''Run all fairness model inside peak and off peak scenario (one episode)'''
    results = {}

    for label, (models_dir, scenario) in FAIRNESS_MODELS.items():
        print(f'running {label}')
        env = get_env(scenario, reward_fn="local")
        agents = load_agents(env, models_dir)

        env.seed = 0
        _, starvation_history = run_episode_extended(env, agents)
        env.close()

        results[label] = starvation_history

    return results

def plot_starvation_timeseries(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, j in enumerate(INTERSECTIONS):
        ax = axes[idx]
        for label, starvation_history in results.items():
            ax.plot(
                starvation_history[j],
                label=label,
                color=MODEL_COLORS[label],
                linewidth=1.2,
                alpha=0.85,
            )
        ax.set_title(f"Intersection {j}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Max Red Time (s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase Starvation Over Time (Fairness Models)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(STARVATION_DIR, exist_ok=True)
    path = os.path.join(STARVATION_DIR, "starvation_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: starvation_timeseries.png")
    plt.close(fig)

def main():
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(STARVATION_DIR, exist_ok=True)

    # uncomment below if needed heatmap, rn i only need phase starvation
    # run_heatmap_analysis()
    starvation_data = collect_starvation_data()
    plot_starvation_timeseries(starvation_data)

if __name__ == '__main__':
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    main()





