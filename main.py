
import os
import argparse
import numpy as np
import torch
import random
import json
from datetime import datetime


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
        choices=['local', 'cooperative', 'fairness'],
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
    print(f"  TrafficMind Training Shuru!")
    print(f"  Reward   : {args.reward}")
    print(f"  Scenario : {args.scenario}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print("=" * 60)

    set_seed(args.seed)

    config_path = get_config_path(args.scenario)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=args.reward,
        use_gui=args.gui
    )

    intersections = ['J1', 'J2', 'J4', 'J5']

    agents = {}
    buffers = {}

    for junction in intersections:
        agents[junction] = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.990,
            target_update_freq=100
        )
        buffers[junction] = ReplayBuffer(capacity=50000)
    
    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        for junction in intersections:
            model_path = os.path.join(args.resume, f"agent_{junction}_final.pth")
            if os.path.exists(model_path):
                agents[junction].load(model_path)
                print(f"  Loaded model for {junction} from {model_path}")
            else:
                print(f"  Warning: Model file not found for {junction} at {model_path}")
        resumed_epsilon = max(0.01, 1.0 * (0.990 ** (args.start_episode)))
        for junction in intersections:
            agents[junction].epsilon = resumed_epsilon
        print(f'epsilon set to : {resumed_epsilon:.3f}')

    episode_rewards = []
    episode_losses = []
    if args.resume:
        output_dir = args.resume
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            'outputs',
            f"{args.reward}_{args.scenario}_{timestamp}"
            )
    os.makedirs(output_dir, exist_ok=True)

    start = args.start_episode if args.resume else 1
    for episode in range(start, args.episodes + 1):

        states = env.reset()
        phase_changes = {j: 0 for j in intersections}
        prev_phases = {j: env._current_phase[j] for j in intersections}
        total_rewards = {j: 0.0 for j in intersections}
        total_losses = []
        step_count = 0
        done = False

        while not done:

            actions = {}
            for junction in intersections:
                actions[junction] = agents[junction].select_action(
                    states[junction]
                )

            next_states, rewards, done = env.step(actions)

            for junction in intersections:
                buffers[junction].push(
                    state=states[junction],
                    action=actions[junction],
                    reward=rewards[junction],
                    next_state=next_states[junction],
                    done=float(done)
                )
                total_rewards[junction] += rewards[junction]

            for junction in intersections:
                loss = agents[junction].train_step(
                    buffers[junction], batch_size=64
                )
                if loss is not None:
                    total_losses.append(loss)

            states = next_states
            step_count += 1

            for junction in intersections:
                if env._current_phase[junction] != prev_phases[junction]:
                    phase_changes[junction] += 1
                prev_phases[junction] = env._current_phase[junction]

        for junction in intersections:
            agents[junction].decay_epsilon()

        avg_reward = np.mean(list(total_rewards.values()))
        avg_loss = np.mean(total_losses) if total_losses else 0.0

        episode_rewards.append(avg_reward)
        episode_losses.append(avg_loss)

        current_epsilon = agents['J1'].epsilon

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d}/{args.episodes} | "
                f"Reward: {avg_reward:8.2f} | "
                f"Loss: {avg_loss:7.4f} | "
                f"Epsilon: {current_epsilon:.3f} | "
                f"Steps: {step_count}"
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
        'episodes': args.episodes,
        'seed': args.seed,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
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