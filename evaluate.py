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
import traci
from dotenv import load_dotenv
 
load_dotenv()
sys.path.append(os.path.join(os.getenv("Sumo_Home", ""), "tools"))
 
from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent
 
INTERSECTIONS = ["J1", "J2", "J4", "J5"]
POLICIES = ["trained_dqn", "fixed_time", "random_legal", "greedy_queue"]
KPI_NAMES = [
    "mean_waiting_time",
    "total_waiting_time",
    "mean_queue_length",
    "max_lane_wait",
    "throughput",
    "switch_count_total",
    "mean_reward_per_step",
]
 
def get_current_green_lanes(env: SumoEnvironment, junction: str) -> set:
    """
    Return the incoming lanes that currently have green signal at this junction.
    This uses SUMO's actual traffic light state instead of hard-coded lane indices.
    """
    green_lanes = set()

    signal_state = traci.trafficlight.getRedYellowGreenState(junction)
    controlled_links = traci.trafficlight.getControlledLinks(junction)

    for signal_index, signal_char in enumerate(signal_state):
        if signal_char not in ("g", "G"):
            continue

        for link in controlled_links[signal_index]:
            incoming_lane = link[0]

            if incoming_lane in env.lanes[junction]:
                green_lanes.add(incoming_lane)

    return green_lanes
 
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
        choices=["local", "cooperative", "fairness"],
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
        default="evaluation_results.json",
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
    """Match action masking in DQNAgent.select_action."""
    phase_time = state[9]
    is_yellow = state[10]
    return is_yellow == 0.0 and phase_time >= 1.0
 
 
def new_kpi_tracker(env: SumoEnvironment) -> Dict:
    num_lanes = sum(len(env.lanes[j]) for j in env.intersections)
    return {
        "total_waiting_time": 0.0,
        "total_queue": 0.0,
        "max_lane_wait": 0.0,
        "step_count": 0,
        "reward_sum": 0.0,
        "switch_count": 0,
        "throughput": 0.0,
        "num_lanes": num_lanes,
    }
 
 
def update_kpi_tracker(
    tracker: Dict,
    env: SumoEnvironment,
    rewards: Dict[str, float],
    executed_actions: Dict[str, int],
) -> None:
    step_wait = 0.0
    step_queue = 0.0
    for junction in env.intersections:
        for lane in env.lanes[junction]:
            wait = traci.lane.getWaitingTime(lane)
            step_wait += wait
            tracker["max_lane_wait"] = max(tracker["max_lane_wait"], wait)
            step_queue += traci.lane.getLastStepHaltingNumber(lane)
 
    tracker["total_waiting_time"] += step_wait
    tracker["total_queue"] += step_queue
    tracker["step_count"] += 1
    tracker["reward_sum"] += float(np.mean(list(rewards.values())))
    tracker["switch_count"] += int(sum(executed_actions.values()))
    tracker["throughput"] += traci.simulation.getArrivedNumber()
 
 
def get_kpis(tracker: Dict) -> Dict[str, float]:
    """Finalize episode KPIs from per-step accumulators."""
    steps = max(tracker["step_count"], 1)
    num_lanes = tracker["num_lanes"]
    lane_steps = steps * num_lanes
 
   
 
    return {
        "mean_waiting_time": tracker["total_waiting_time"] / lane_steps,
        "total_waiting_time": tracker["total_waiting_time"],
        "mean_queue_length": tracker["total_queue"] / lane_steps,
        "max_lane_wait": tracker["max_lane_wait"],
        "throughput": float(tracker["throughput"]),
        "switch_count_total": float(tracker["switch_count"]),
        "mean_reward_per_step": tracker["reward_sum"] / steps,
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
            target_update_freq=50,
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

    green_lanes = get_current_green_lanes(env, junction)

    if not green_lanes:
        return 0

    all_lanes = set(env.lanes[junction])
    red_lanes = all_lanes - green_lanes

    current_q = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in green_lanes
    )

    opposite_q = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in red_lanes
    )

    return 1 if opposite_q > current_q else 0
 
def make_policy_fn(
    policy_name: str,
    env: SumoEnvironment,
    agents: Optional[Dict[str, DQNAgent]] = None,
) -> Callable[[Dict[str, np.ndarray]], Dict[str, int]]:
    if policy_name == "trained_dqn":
        if agents is None:
            raise ValueError("trained_dqn requires loaded agents")
 
        def select(states: Dict[str, np.ndarray]) -> Dict[str, int]:
            return {
                j: agents[j].select_action(states[j]) for j in INTERSECTIONS
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
    tracker = new_kpi_tracker(env)
    done = False
 
    while not done:
        actions = select_actions(states)
        states, rewards, done, executed = env.step(actions)
        update_kpi_tracker(tracker, env, rewards, executed)
 
    return get_kpis(tracker)
 
 
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
        select_actions = make_policy_fn(policy_name, env, agents)
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
    args = parse_args()
    results = evaluate(args)
 
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
 
    print(f"  Results saved: {output_path}")
    print_comparison_table(results)
 
 
if __name__ == "__main__":
    main()
 