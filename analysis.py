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
    "Local Peak":                    "#4C9BE8",
    "Cooperative Peak":              "#E8774C",
    "Fairness Peak":                 "#4CE89B",
    "Pressure Local Peak":           "#9BC7F5",
    "Pressure Cooperative Peak":     "#F5B39B",
    "Pressure Fairness Peak":        "#9BF5C7",
    "Local Off-Peak":                "#1A5FA8",
    "Cooperative Off-Peak":          "#A83A10",
    "Fairness Off-Peak":             "#1A9B5F",
    "Pressure Local Off-Peak":       "#5F8FD6",
    "Pressure Cooperative Off-Peak": "#D67A5F",
    "Pressure Fairness Off-Peak":    "#5FD69B",
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

def run_episode_extended(env, agents):
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
        wait_history, _ = run_episode_extended(env, agents)
        env.close()

        means = {j: float(np.mean(wait_history[j])) for j in INTERSECTIONS}
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

def main():
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(STARVATION_DIR, exist_ok=True)

    run_heatmap_analysis()

if __name__ == '__main__':
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    main()





