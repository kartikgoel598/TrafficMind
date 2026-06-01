import os
import argparse
import json
import random
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate TrafficMind policies (greedy RL vs fixed-time baseline)'
    )
    parser.add_argument('--reward', type=str, default='local', choices=['local', 'cooperative', 'fairness'])
    parser.add_argument('--scenario', type=str, default='peak', choices=['peak', 'off_peak'])
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Folder containing agent_<J>_final.pth')
    parser.add_argument('--episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for deterministic evaluation')
    parser.add_argument('--policy', type=str, default='both', choices=['rl', 'fixed', 'both'])
    parser.add_argument('--gui', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_config_path(scenario):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if scenario == 'peak':
        return os.path.join(base_dir, 'sumo', 'configs', 'peak.sumocfg')
    return os.path.join(base_dir, 'sumo', 'configs', 'off_peak.sumocfg')


def build_agents(env, checkpoint_dir):
    intersections = ['J1', 'J2', 'J4', 'J5']
    agents = {}
    for junction in intersections:
        model_path = os.path.join(checkpoint_dir, f'agent_{junction}_final.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Missing checkpoint: {model_path}')
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
        agent.load(model_path)
        agent.epsilon = 0.0  # Greedy policy for valid evaluation.
        agents[junction] = agent
    return agents


def fixed_time_actions(env):
    actions = {}
    for junction in env.intersections:
        # Baseline with same environment constraints:
        # switch whenever min_green is satisfied and no yellow is active.
        can_switch = env._yellow_timer[junction] == 0 and env._phase_time[junction] >= env.min_green_time
        actions[junction] = 1 if can_switch else 0
    return actions


def aggregate_episode_metrics(step_sums, step_count):
    return {
        'mean_waiting_time': step_sums['mean_waiting_time'] / max(1, step_count),
        'total_waiting_time': step_sums['total_waiting_time'] / max(1, step_count),
        'mean_queue_length': step_sums['mean_queue_length'] / max(1, step_count),
        'max_lane_wait': step_sums['max_lane_wait'],
        'throughput': step_sums['throughput'],
        'step_count': step_count,
        'switch_count': step_sums['switch_count'],
    }


def run_single_episode(env, mode, agents=None):
    states = env.reset()
    done = False
    step_count = 0
    step_sums = {
        'mean_waiting_time': 0.0,
        'total_waiting_time': 0.0,
        'mean_queue_length': 0.0,
        'max_lane_wait': 0.0,
        'throughput': 0,
        'switch_count': 0,
    }

    while not done:
        if mode == 'rl':
            actions = {j: agents[j].select_action(states[j]) for j in env.intersections}
        else:
            actions = fixed_time_actions(env)

        next_states, _, done, executed_actions = env.step(actions)
        kpi = env.get_kpis()
        step_sums['mean_waiting_time'] += kpi['mean_waiting_time']
        step_sums['total_waiting_time'] += kpi['total_waiting_time']
        step_sums['mean_queue_length'] += kpi['mean_queue_length']
        step_sums['max_lane_wait'] = max(step_sums['max_lane_wait'], kpi['max_lane_wait'])
        step_sums['throughput'] += kpi['throughput_step']
        step_sums['switch_count'] += sum(int(executed_actions[j] == 1) for j in env.intersections)
        step_count += 1
        states = next_states

    return aggregate_episode_metrics(step_sums, step_count)


def summarize_metrics(episodes):
    keys = ['mean_waiting_time', 'total_waiting_time', 'mean_queue_length', 'max_lane_wait', 'throughput', 'switch_count']
    summary = {}
    for k in keys:
        vals = [e[k] for e in episodes]
        summary[k] = {
            'mean': float(np.mean(vals)) if vals else 0.0,
            'std': float(np.std(vals)) if vals else 0.0,
        }
    return summary


def evaluate_mode(args, mode, agents=None):
    env = SumoEnvironment(
        config_path=get_config_path(args.scenario),
        reward_fn=args.reward,
        use_gui=args.gui,
        seed=args.seed,
    )
    episode_metrics = []
    for ep in range(1, args.episodes + 1):
        # Deterministic seed schedule for reproducible comparisons.
        env.seed = args.seed + ep
        metrics = run_single_episode(env, mode=mode, agents=agents)
        episode_metrics.append(metrics)
        print(
            f"[{mode}] ep {ep:02d}/{args.episodes} | "
            f"wait={metrics['mean_waiting_time']:.2f} | "
            f"queue={metrics['mean_queue_length']:.2f} | "
            f"throughput={metrics['throughput']}"
        )
    env.close()
    return {
        'mode': mode,
        'episodes': episode_metrics,
        'summary': summarize_metrics(episode_metrics),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    results = {
        'reward_fn': args.reward,
        'scenario': args.scenario,
        'episodes': args.episodes,
        'seed': args.seed,
        'evaluations': [],
    }

    if args.policy in ('rl', 'both'):
        env_for_shapes = SumoEnvironment(get_config_path(args.scenario), reward_fn=args.reward, use_gui=args.gui, seed=args.seed)
        agents = build_agents(env_for_shapes, args.checkpoint_dir)
        env_for_shapes.close()
        results['evaluations'].append(evaluate_mode(args, mode='rl', agents=agents))

    if args.policy in ('fixed', 'both'):
        results['evaluations'].append(evaluate_mode(args, mode='fixed'))

    output_path = os.path.join(args.checkpoint_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to: {output_path}")


if __name__ == '__main__':
    main()
