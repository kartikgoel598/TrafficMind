"""
Evaluate trained DQN agents and baseline policies on SUMO scenarios.
No training — inference / rule-based control only.
"""

import os
import sys
import argparse
import json
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from utils.traffic_signal_utils import get_green_red_queues
from utils.webster_utils import compute_webster_timing

load_dotenv()
sumo_home = os.getenv("Sumo_Home")
if not sumo_home:
    raise EnvironmentError(
        "Sumo_Home is not set. Add it to your .env file (see README)."
    )
sys.path.append(os.path.join(sumo_home, "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent

INTERSECTIONS = ["J1", "J2", "J4", "J5"]
POLICIES = ["trained_dqn", "fixed_time", "random_legal", "greedy_queue", "webster_static"]
KPI_NAMES = [
    "mean_waiting_time",
    "total_waiting_time",
    "mean_queue_length",
    "max_lane_wait",
    "throughput",
    "switch_count_total",
    "mean_reward_per_step",
    "step_count",
]
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="TrafficMind — evaluate DQN vs baselines"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory with agent_J1_final.pth, ...",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="local",
        choices=["local", "cooperative", "fairness", "pressure_local",'pressure_cooperative','pressure_fairness'],
        help="Reward function used during training (for env rewards)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="peak",
        choices=["peak", "off_peak"],
        help="Traffic scenario",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1000],
        help="Evaluation seeds (one episode each)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable SUMO GUI",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluations/evaluation_results.json",
        help="Path for JSON results",
    )
    return parser.parse_args()


def get_config_path(scenario: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = {
        "peak": os.path.join(base_dir, "sumo", "configs", "peak.sumocfg"),
        "off_peak": os.path.join(base_dir, "sumo", "configs", "off_peak.sumocfg"),
    }
    return configs[scenario]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def is_switch_legal(state: np.ndarray) -> bool:
    phase_time = state[9]
    is_yellow = state[10]
    return is_yellow < 0.5 and phase_time >= 1.0


def aggregate_episode_kpis(
    kpi_sum: Dict[str, float],
    step_count: int,
    switch_count: int,
    reward_sum: float,
) -> Dict[str, float]:
    """Match episode KPI aggregation in main.py."""
    steps = max(step_count, 1)
    return {
        "mean_waiting_time": kpi_sum["mean_waiting_time"] / steps,
        "total_waiting_time": kpi_sum["total_waiting_time"] / steps,
        "mean_queue_length": kpi_sum["mean_queue_length"] / steps,
        "max_lane_wait": kpi_sum["max_lane_wait"],
        "throughput": kpi_sum["throughput"],
        "switch_count_total": float(switch_count),
        "mean_reward_per_step": reward_sum / steps,
        "step_count": float(step_count),
    }


def new_kpi_sum() -> Dict[str, float]:
    return {
        "mean_waiting_time": 0.0,
        "total_waiting_time": 0.0,
        "mean_queue_length": 0.0,
        "max_lane_wait": 0.0,
        "throughput": 0.0,
    }


def load_agents(env: SumoEnvironment, models_dir: str) -> Dict[str, DQNAgent]:
    agents = {}
    for junction in INTERSECTIONS:
        path = os.path.join(models_dir, f"agent_{junction}_final.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing model: {path}")
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            lr=0.0005,
            gamma=0.99,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            target_update_freq=500,
        )
        agent.load(path)
        agent.epsilon = 0.0
        agents[junction] = agent
    return agents


def fixed_time_action(env: SumoEnvironment, junction: str, _state: np.ndarray) -> int:
    if env._yellow_timer[junction] > 0:
        return 0
    if env._phase_time[junction] >= env.min_green_time:
        return 1
    return 0


def random_legal_action(_env: SumoEnvironment, _junction: str, state: np.ndarray) -> int:
    if not is_switch_legal(state):
        return 0
    return random.randint(0, 1)

def greedy_queue_action(
    env: SumoEnvironment, junction: str, state: np.ndarray
) -> int:
    if not is_switch_legal(state):
        return 0

    green_queue, red_queue = get_green_red_queues(env, junction)

    return 1 if red_queue > green_queue else 0

def webster_static_action(
    env: SumoEnvironment,
    junction: str,
    state: np.ndarray,
    timing: Dict,
) -> int:
    """
    Static Webster baseline.
    Switches after the precomputed Webster green time for the current phase.
    """

    if env._yellow_timer[junction] > 0:
        return 0

    current_phase = env._current_phase[junction]

    if current_phase == 0:
        green_limit = timing["green0"]
    else:
        green_limit = timing["green1"]

    if env._phase_time[junction] >= green_limit:
        return 1

    return 0

def make_policy_fn(
    policy_name: str,
    env: SumoEnvironment,
    agents: Optional[Dict[str, DQNAgent]] = None,
    webster_timing: Optional[Dict] = None,
) -> Callable[[Dict[str, np.ndarray]], Dict[str, int]]:
    if policy_name == "trained_dqn":
        if agents is None:
            raise ValueError("trained_dqn requires loaded agents")

        def select(states: Dict[str, np.ndarray]) -> Dict[str, int]:
            return {
                j: agents[j].select_action(states[j]) for j in INTERSECTIONS
            }

        return select
    if policy_name == "webster_static":
        if webster_timing is None:
            raise ValueError("webster_static requires webster_timing")
        def select(states: Dict[str, np.ndarray]) -> Dict[str, int]:
            return {
                j: webster_static_action(env, j, states[j], webster_timing) for j in INTERSECTIONS
            }
        return select

    baselines = {
        "fixed_time": fixed_time_action,
        "random_legal": random_legal_action,
        "greedy_queue": greedy_queue_action,
    }
    baseline_fn = baselines[policy_name]

    def select(states: Dict[str, np.ndarray]) -> Dict[str, int]:
        return {j: baseline_fn(env, j, states[j]) for j in INTERSECTIONS}

    return select


def run_episode(
    env: SumoEnvironment,
    select_actions: Callable[[Dict[str, np.ndarray]], Dict[str, int]],
    seed: int,
) -> Dict[str, float]:
    env.seed = seed
    states = env.reset()
    kpi_sum = new_kpi_sum()
    reward_sum = 0.0
    switch_count = 0
    step_count = 0
    done = False

    while not done:
        actions = select_actions(states)
        states, rewards, done, executed = env.step(actions)
        step_kpis = env.get_kpis()

        kpi_sum["mean_waiting_time"] += step_kpis["mean_waiting_time"]
        kpi_sum["total_waiting_time"] += step_kpis["total_waiting_time"]
        kpi_sum["mean_queue_length"] += step_kpis["mean_queue_length"]
        kpi_sum["max_lane_wait"] = max(
            kpi_sum["max_lane_wait"], step_kpis["max_lane_wait"]
        )
        kpi_sum["throughput"] += step_kpis["throughput_step"]

        reward_sum += float(np.mean(list(rewards.values())))
        switch_count += int(sum(executed.values()))
        step_count += 1

    return aggregate_episode_kpis(kpi_sum, step_count, switch_count, reward_sum)


def aggregate_policy_results(
    per_seed_kpis: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    agg = {}
    for name in KPI_NAMES:
        values = [k[name] for k in per_seed_kpis]
        agg[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return agg


def print_comparison_table(results: Dict) -> None:
    policy_stats = {
        p: results["policies"][p]["aggregate"] for p in POLICIES
    }

    col_w = 22
    header = f"{'KPI':<28}" + "".join(f"{p:>{col_w}}" for p in POLICIES)
    print("\n" + "=" * len(header))
    print("  Evaluation comparison (mean ± std across seeds)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for kpi in KPI_NAMES:
        row = f"{kpi:<28}"
        for policy in POLICIES:
            stats = policy_stats[policy][kpi]
            cell = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print("=" * len(header) + "\n")


def evaluate(args) -> Dict:
    set_seed(args.seeds[0])
    config_path = get_config_path(args.scenario)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=args.reward,
        use_gui=args.gui,
        seed=args.seeds[0],
    )

    if args.scenario == "peak":
        phase0_flow = 400.0
        phase1_flow = 300.0
    else:
        phase0_flow = 160.0
        phase1_flow = 120.0
    webster_timing = compute_webster_timing(
        phase0_flow=phase0_flow,
        phase1_flow=phase1_flow,
        saturation_flow=1800.0,
        yellow_time=env.yellow_duration,
        min_green=env.min_green_time,
        min_cycle=36.0,
        max_cycle=120.0,
    )

    print(f"  Webster : {webster_timing}")

    agents = load_agents(env, args.models_dir)

    results = {
        "models_dir": os.path.abspath(args.models_dir),
        "reward_fn": args.reward,
        "scenario": args.scenario,
        "seeds": args.seeds,
        "policies": {},
    }

    print("=" * 60)
    print("  TrafficMind Evaluation")
    print(f"  Models   : {args.models_dir}")
    print(f"  Reward   : {args.reward}")
    print(f"  Scenario : {args.scenario}")
    print(f"  Seeds    : {args.seeds}")
    print("=" * 60)

    for policy_name in POLICIES:
        print(f"\n  Policy: {policy_name}")
        select_actions = make_policy_fn(policy_name, env, agents,webster_timing = webster_timing)
        per_seed = []

        for seed in args.seeds:
            set_seed(seed)
            kpis = run_episode(env, select_actions, seed)
            per_seed.append(kpis)
            print(
                f"    seed {seed:4d} | "
                f"wait={kpis['mean_waiting_time']:.2f} | "
                f"queue={kpis['mean_queue_length']:.2f} | "
                f"throughput={kpis['throughput']:.0f} | "
                f"switches={kpis['switch_count_total']:.0f}"
            )

        results["policies"][policy_name] = {
            "per_seed": {str(args.seeds[i]): per_seed[i] for i in range(len(args.seeds))},
            "aggregate": aggregate_policy_results(per_seed),
        }

    env.close()
    return results


def main():
    print('Starting evaluation...')
    args = parse_args()
    results = evaluate(args)

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {output_path}")
    print_comparison_table(results)


if __name__ == "__main__":
    main()

# python evaluate.py --models-dir "outputs/fairness_peak_final" --scenario peak --reward fairness
# python evaluate.py --models-dir "outputs/fairness_off_peak_final" --scenario off_peak --reward fairness