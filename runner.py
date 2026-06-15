"""
runner.py — single-agent episode runner for the TrafficMind versus dashboard.

Runs one policy for one episode (single seed) and writes live stats to a JSON
file every simulation step so the dashboard can display them in real time.

Usage (called by server.py, not directly):
    python runner.py --agent trained_dqn --models-dir outputs/local_peak
                     --reward local --scenario peak --seed 42
                     --live-file live_a.json --slot a
"""

import os, sys, argparse, json, random, time
from typing import Dict, Optional
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
sumo_home = os.getenv("Sumo_Home")
if not sumo_home:
    raise EnvironmentError("Sumo_Home not set in .env")
sys.path.append(os.path.join(sumo_home, "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent
from utils.traffic_signal_utils import get_green_red_queues
from utils.webster_utils import compute_webster_timing

INTERSECTIONS = ["J1", "J2", "J4", "J5"]

# ── Policies (copied from evaluate.py) ────────────────────────────────────────

FIXED_TIME_GREEN = 20

def is_switch_legal(state: np.ndarray) -> bool:
    return state[10] < 0.5 and state[9] >= 1.0

def fixed_time_action(env, junction, _state):
    if env._yellow_timer[junction] > 0:
        return 0
    if env._phase_time[junction] >= FIXED_TIME_GREEN:
        return 1
    return 0

def random_legal_action(_env, _junction, state):
    if not is_switch_legal(state):
        return 0
    return random.randint(0, 1)

def greedy_queue_action(env, junction, state):
    if not is_switch_legal(state):
        return 0
    green_q, red_q = get_green_red_queues(env, junction)
    return 1 if red_q > green_q else 0

def webster_static_action(env, junction, state, timing):
    if env._yellow_timer[junction] > 0:
        return 0
    phase = env._current_phase[junction]
    green_limit = timing["green0"] if phase == 0 else timing["green1"]
    if env._phase_time[junction] >= green_limit:
        return 1
    return 0

def load_dqn_agents(env, models_dir):
    agents = {}
    for j in INTERSECTIONS:
        path = os.path.join(models_dir, f"agent_{j}_final.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing model: {path}")
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            lr=0.0005, gamma=0.99,
            epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0,
            target_update_freq=500,
        )
        agent.load(path)
        agent.epsilon = 0.0
        agents[j] = agent
    return agents

def make_select_fn(policy_name, env, agents=None, webster_timing=None):
    if policy_name == "trained_dqn":
        def select(states):
            return {j: agents[j].select_action(states[j]) for j in INTERSECTIONS}
        return select

    if policy_name == "webster_static":
        def select(states):
            return {j: webster_static_action(env, j, states[j], webster_timing) for j in INTERSECTIONS}
        return select

    baseline_map = {
        "fixed_time":    fixed_time_action,
        "random_legal":  random_legal_action,
        "greedy_queue":  greedy_queue_action,
    }
    fn = baseline_map[policy_name]
    def select(states):
        return {j: fn(env, j, states[j]) for j in INTERSECTIONS}
    return select

# ── Live stats writer ──────────────────────────────────────────────────────────

def write_live(path: str, payload: dict):
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass  # skip if server is reading at this exact moment

# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",       required=True,
                   choices=["trained_dqn","fixed_time","random_legal",
                            "greedy_queue","webster_static"])
    p.add_argument("--models-dir",  default=None)
    p.add_argument("--reward",      default="local")
    p.add_argument("--scenario",    default="peak")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--live-file",   default="live_a.json",
                   help="Path to write live stats JSON each step")
    p.add_argument("--slot",        default="a",
                   help="Which dashboard slot this is (a or b) — for display only")
    p.add_argument("--gui",         action="store_true", default=True)
    return p.parse_args()


def get_config_path(scenario):
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "peak":     os.path.join(base, "sumo", "configs", "peak.sumocfg"),
        "off_peak": os.path.join(base, "sumo", "configs", "off_peak.sumocfg"),
    }[scenario]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = get_config_path(args.scenario)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=args.reward,
        use_gui=args.gui,
        seed=args.seed,
    )

    # Webster timing (needed even for non-webster policies — cheap to compute)
    if args.scenario == "peak":
        p0_flow, p1_flow = 400.0, 300.0
    else:
        p0_flow, p1_flow = 160.0, 120.0

    webster_timing = compute_webster_timing(
        phase0_flow=p0_flow, phase1_flow=p1_flow,
        saturation_flow=1800.0,
        yellow_time=env.yellow_duration,
        min_green=env.min_green_time,
        min_cycle=36.0, max_cycle=120.0,
    )

    agents = None
    if args.agent == "trained_dqn":
        if not args.models_dir:
            raise ValueError("--models-dir required for trained_dqn")
        agents = load_dqn_agents(env, args.models_dir)

    select = make_select_fn(args.agent, env, agents, webster_timing)

    # ── Write initial "loading" state ─────────────────────────────────────────
    write_live(args.live_file, {
        "status": "running",
        "agent": args.agent,
        "slot": args.slot,
        "seed": args.seed,
        "scenario": args.scenario,
        "step": 0,
        "sim_time": 0.0,
        "mean_waiting_time": 0.0,
        "total_waiting_time": 0.0,
        "mean_queue_length": 0.0,
        "max_lane_wait": 0.0,
        "throughput": 0,
        "switch_count": 0,
    })

    states = env.reset()
    done = False

    from collections import deque
    # Running totals
    wait_window = deque(maxlen=50)
    queue_window = deque(maxlen=50)
    wait_total_window = deque(maxlen=50)
    total_waiting_time = 0.0
    total_throughput   = 0
    switch_count       = 0
    step_count         = 0
    max_lane_wait_seen = 0.0


    while not done:
        actions = select(states)
        states, rewards, done, executed = env.step(actions)
        kpis = env.get_kpis()

        wait_window.append(kpis['mean_waiting_time'])
        queue_window.append(kpis['mean_queue_length'])
        wait_total_window.append(kpis['total_waiting_time'])

        step_count         += 1
        switch_count       += int(sum(executed.values()))
        total_throughput   += kpis["throughput_step"]

        write_live(args.live_file, {
            "status": "running",
            "agent": args.agent,
            "slot": args.slot,
            "seed": args.seed,
            "scenario": args.scenario,
            "step": step_count,
            "sim_time": kpis["sim_time"],
            # live per-step metrics
            "mean_waiting_time":    round(float(np.mean(wait_window)), 2),
            "mean_queue_length":    round(float(np.mean(queue_window)), 2),
            "total_waiting_time":   round(float(np.mean(wait_total_window)), 2),
            "max_lane_wait":        round(kpis["max_lane_wait"], 2),
            "throughput":           total_throughput,
            "switch_count":         switch_count,
        })

    # ── Final state ───────────────────────────────────────────────────────────
    write_live(args.live_file, {
        "status": "done",
        "agent": args.agent,
        "slot": args.slot,
        "seed": args.seed,
        "scenario": args.scenario,
        "step": step_count,
        "sim_time": 0.0,
        "mean_waiting_time":   0.0,
        "total_waiting_time":  round(total_waiting_time / max(1, step_count), 2),
        "mean_queue_length":   0.0,
        "max_lane_wait":       round(max_lane_wait_seen, 2),
        "throughput":          total_throughput,
        "switch_count":        switch_count,
    })

    env.close()
    print(f"[runner:{args.slot}] done — {step_count} steps, "
          f"throughput={total_throughput}, switches={switch_count}")


if __name__ == "__main__":
    main()