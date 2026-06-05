import os
import argparse
import numpy as np
import torch
import random
import json
from datetime import datetime
import sys

from dotenv import load_dotenv
load_dotenv()  # loads Sumo_Home from .env before anything else runs
sys.path.append(os.path.join(os.getenv('Sumo_Home'), "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser(
        description='TrafficMind — DQN Traffic Signal Control'
    )

    parser.add_argument(
        '--reward',
        type=str,
        default='local',
        choices=['local', 'cooperative', 'fairness','pressure_local'],
        help='Reward function: local, cooperative, or fairness'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='peak',
        choices=['peak', 'off_peak'],
        help='Traffic scenario: peak or off_peak'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='how much epiosdes you want to train'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='SUMO GUI on or off'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a checkpoint to resume training'
    )

    parser.add_argument(
        '--start_episode',
        type=int,
        default=1,
        help='Episode to start training from (default: 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Replay sample batch size (default: 128)'
    )
    parser.add_argument(
        '--replay_warmup',
        type=int,
        default=1000,
        help='Minimum transitions per agent before training starts'
    )
    parser.add_argument(
        '--randomize_episode_seed',
        action='store_true',
        help='Use fully random SUMO seed per episode (higher variance)'
    )

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_config_path(scenario):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    configs = {
        'peak'    : os.path.join(base_dir, 'sumo', 'configs', 'peak.sumocfg'),
        'off_peak': os.path.join(base_dir, 'sumo', 'configs', 'off_peak.sumocfg'),
    }
    return configs[scenario]


def train(args):
    print("=" * 60)
    print(f"  TrafficMind Training started!")
    print(f"  Reward   : {args.reward}")
    print(f"  Scenario : {args.scenario}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Warmup   : {args.replay_warmup}")
    print("=" * 60)

    set_seed(args.seed)

    config_path = get_config_path(args.scenario)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=args.reward,
        use_gui=args.gui,
        seed=args.seed
    )

    intersections = ['J1', 'J2', 'J4', 'J5']

    agents = {}
    buffers = {}

    for junction in intersections:
        agents[junction] = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            lr=0.0005,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.990,
            target_update_freq=500
        )
        buffers[junction] = ReplayBuffer(capacity=100000)
    
    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        for junction in intersections:
            model_path = os.path.join(args.resume, f"agent_{junction}_final.pth")
            if os.path.exists(model_path):
                agents[junction].load(model_path)
                print(f"  Loaded model for {junction} from {model_path}")
            else:
                print(f"  Warning: Model file not found for {junction} at {model_path}")
        print(f'loaded epsilon from checkpoint: {agents['J1'].epsilon:.3f}')
        print('target networks loaded from checkpoint')

    
    if args.resume:
        output_dir = args.resume
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            'outputs',
            f"{args.reward}_{args.scenario}_tu500_lr0005_{timestamp}"
            )
    os.makedirs(output_dir, exist_ok=True)

    episode_rewards = []
    episode_rewards_per_step = []
    episode_losses = []
    episode_kpis = []
    episode_indices = []
    if args.resume:
        old_path = os.path.join(args.resume, 'results.json')
        if os.path.exists(old_path):
            with open(old_path, 'r') as f:
                old_results = json.load(f)
            episode_rewards = old_results.get('episode_rewards', [])
            episode_rewards_per_step = old_results.get('episode_rewards_per_step', [])
            episode_losses  = old_results.get('episode_losses', [])
            episode_kpis = old_results.get('episode_kpis', [])
            episode_indices = old_results.get('episode_indices', list(range(1, len(episode_rewards) + 1)))
            print(f"  old {len(episode_rewards)} episodes load")

            # added warning if user forgot to add --start_episode
            actual_episodes = len(episode_rewards)
            if args.start_episode - 1 != actual_episodes:
                print(f"  WARNING: --start_episode {args.start_episode} doesn't match "
                    f"checkpoint history ({actual_episodes} episodes completed)")

    start = args.start_episode if args.resume else 1
    for episode in range(start, args.episodes + 1):
        # Fix: deterministic-by-default episode seed for lower training variance.
        if args.randomize_episode_seed:
            env.seed = random.randint(0, 9999)
        else:
            env.seed = args.seed + episode
        
        states = env.reset()
        phase_changes = {j: 0 for j in intersections}
        prev_phases = {j: env._current_phase[j] for j in intersections}
        total_rewards = {j: 0.0 for j in intersections}
        total_losses = []
        total_switches = {j: 0 for j in intersections}
        legal_switch_opportunities = {j: 0 for j in intersections}
        switch_when_legal = {j: 0 for j in intersections}
        keep_when_legal = {j: 0 for j in intersections}
        illegal_switch_requests = {j: 0 for j in intersections}
        kpi_sum = {
            'mean_waiting_time': 0.0,
            'total_waiting_time': 0.0,
            'mean_queue_length': 0.0,
            'max_lane_wait': 0.0,
            'throughput': 0,
        }
        step_count = 0
        done = False

        while not done:
            actions = {}
            for junction in intersections:
                actions[junction] = agents[junction].select_action(states[junction])
            for junction in intersections:
                phase_time_ready = env._phase_time[junction] >= env.min_green_time
                not_yellow = env._yellow_timer[junction] ==0
                switch_legal = phase_time_ready and not_yellow
                if switch_legal:
                    legal_switch_opportunities[junction] += 1
                    if actions[junction] == 1:
                        switch_when_legal[junction] += 1
                    else:
                        keep_when_legal[junction] += 1
                else:
                    if actions[junction] == 1:
                        illegal_switch_requests[junction] += 1

            next_states, rewards, done, executed_actions = env.step(actions)

            for junction in intersections:
                buffers[junction].push(
                    state=states[junction],
                    action=executed_actions[junction], # store executed SUMO action
                    reward=rewards[junction],
                    next_state=next_states[junction],
                    done=float(done)
                )
                total_rewards[junction] += rewards[junction]
                total_switches[junction] += int(executed_actions[junction] == 1)

            for junction in intersections:
                # Fix: warmup replay before updates to reduce unstable early Q-targets.
                if len(buffers[junction]) < args.replay_warmup:
                    continue
                loss = agents[junction].train_step(buffers[junction], batch_size=args.batch_size)
                if loss is not None:
                    total_losses.append(loss)

            states = next_states
            step_count += 1
            step_kpis = env.get_kpis()
            kpi_sum['mean_waiting_time'] += step_kpis['mean_waiting_time']
            kpi_sum['total_waiting_time'] += step_kpis['total_waiting_time']
            kpi_sum['mean_queue_length'] += step_kpis['mean_queue_length']
            kpi_sum['max_lane_wait'] = max(kpi_sum['max_lane_wait'], step_kpis['max_lane_wait'])
            kpi_sum['throughput'] += step_kpis['throughput_step']

            for junction in intersections:
                if env._current_phase[junction] != prev_phases[junction]:
                    phase_changes[junction] += 1
                prev_phases[junction] = env._current_phase[junction]

        for junction in intersections:
            agents[junction].decay_epsilon()

        avg_reward = np.mean(list(total_rewards.values()))
        # Fix: this is the primary learning curve to compare runs fairly.
        avg_reward_per_step = np.mean(
            [total_rewards[j] / max(step_count, 1) for j in intersections]
        )
        avg_loss = float(np.mean(total_losses)) if total_losses else None
        episode_kpi = {
            'mean_waiting_time': kpi_sum['mean_waiting_time'] / max(step_count, 1),
            'total_waiting_time': kpi_sum['total_waiting_time'] / max(step_count, 1),
            'mean_queue_length': kpi_sum['mean_queue_length'] / max(step_count, 1),
            'max_lane_wait': kpi_sum['max_lane_wait'],
            'throughput': kpi_sum['throughput'],
            'step_count': step_count,
            'switch_count_total': int(sum(total_switches.values())),
            'switch_count_by_junction': total_switches,
            'phase_changes_by_junction': phase_changes,
            'legal_switch_opportunities': legal_switch_opportunities,
            'switch_when_legal': switch_when_legal,
            'keep_when_legal': keep_when_legal,
            'illegal_switch_requests': illegal_switch_requests,

        }

        episode_rewards.append(avg_reward)
        episode_rewards_per_step.append(avg_reward_per_step)
        episode_losses.append(avg_loss)
        episode_kpis.append(episode_kpi)
        episode_indices.append(episode)

        current_epsilon = agents['J1'].epsilon

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d}/{args.episodes} | "
                f"Reward: {avg_reward:8.2f} | "
                f"R/Step: {avg_reward_per_step:8.4f} | "
                f"Loss: {avg_loss:7.4f} | "
                f"Epsilon: {current_epsilon:.3f} | "
                f"Steps: {step_count} | "
                f"Wait: {episode_kpi['mean_waiting_time']:.2f} | "
                f"Thrpt: {episode_kpi['throughput']}"
            )
            

        if episode % 100 == 0:
            for junction in intersections:
                model_path = os.path.join(
                    output_dir,
                    f"agent_{junction}_ep{episode}.pth"
                )
                agents[junction].save(model_path)
            print(f"  → Models saved at episode {episode}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    for junction in intersections:
        model_path = os.path.join(
            output_dir, f"agent_{junction}_final.pth"
        )
        agents[junction].save(model_path)

    results = {
        'reward_fn': args.reward,
        'scenario': args.scenario,
        'episodes': len(episode_rewards),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'replay_warmup': args.replay_warmup,
        'randomize_episode_seed': args.randomize_episode_seed,
        'episode_indices': episode_indices,
        'episode_rewards': episode_rewards,
        'episode_rewards_per_step': episode_rewards_per_step,
        'episode_losses': episode_losses,
        'episode_kpis': episode_kpis,
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {results_path}")
    print(f"  Models saved : {output_dir}")

    env.close()

    return results

if __name__ == '__main__':
    args = parse_args()
    train(args)
