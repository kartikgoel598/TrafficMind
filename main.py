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
        description='TrafficMind - DQN Traffic Signal Control'
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

    return parser.parse_args()

def load_checkpoint_meta(resume_path: str):
    results_path = os.path.join(resume_path, 'results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f'No results.json found in {resume_path}')
    with open(results_path, 'r') as f:
        return json.load(f)

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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    
    episode_rewards = []
    episode_losses = []
    start_episode = 1
    if args.resume:
        meta = load_checkpoint_meta(args.resume)
        reward_fn       = meta['reward_fn']
        scenario        = meta['scenario']
        episode_rewards = meta.get('episode_rewards', [])
        episode_losses  = meta.get('episode_losses',  [])
        start_episode   = len(episode_rewards) + 1
        output_dir      = args.resume
        if start_episode > args.episodes:
            print(f"  Nothing to do — checkpoint already has {len(episode_rewards)} episodes.")
            return
    else:
        reward_fn  = args.reward
        scenario   = args.scenario
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', f"{reward_fn}_{scenario}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  TrafficMind Training started!")
    print(f"  Reward   : {reward_fn}")
    print(f"  Scenario : {scenario}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print("=" * 60)

    set_seed(args.seed)

    config_path = get_config_path(scenario)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=reward_fn,
        use_gui=args.gui,
        seed=args.seed
    )

    # highen the learning starts 
    LEARNING_STARTS = 5000
    EPSILON_DECAY = 0.999988
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
            epsilon_decay=EPSILON_DECAY,
            target_update_freq=500
        )
        buffers[junction] = ReplayBuffer(capacity=100000)


    if args.resume:
        for junction in intersections:
            model_path = os.path.join(args.resume, f"agent_{junction}_final.pth")
            if os.path.exists(model_path):
                agents[junction].load(model_path)
            else:
                print(f"  WARNING: model not found for {junction}")
        steps_done = (start_episode - 1) * 900
        resumed_epsilon = max(0.01, EPSILON_DECAY ** steps_done)
        for junction in intersections:
            agents[junction].epsilon = resumed_epsilon
            agents[junction]._update_target_network()


        print(f"  Epsilon restored to {resumed_epsilon:.4f}")


     # the actual training
    for episode in range(start_episode, args.episodes + 1):
        env.seed = random.randint(0, 9999)
        
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
                actions[junction] = agents[junction].select_action(states[junction])

            next_states, rewards, done, executed_actions = env.step(actions)


            for junction in intersections:
                buffers[junction].push(
                    state=states[junction],
                    action=executed_actions[junction], # change to actually executed action in SUMO to memory (buffer)
                    reward=rewards[junction],
                    next_state=next_states[junction],
                    done=float(done)
                )
                total_rewards[junction] += rewards[junction]

            if step_count % 4 == 0:
                for junction in intersections:
                    if len(buffers[junction]) >= LEARNING_STARTS:
                        loss = agents[junction].train_step(
                            buffers[junction], batch_size=128
                        )

                        if loss is not None:
                            total_losses.append(loss)

            new_epsilon = max(0.01, agents['J1'].epsilon * agents['J1'].epsilon_decay)
            for junction in intersections:
                agents[junction].epsilon = new_epsilon

            states = next_states
            step_count += 1

            for junction in intersections:
                if env._current_phase[junction] != prev_phases[junction]:
                    phase_changes[junction] += 1
                prev_phases[junction] = env._current_phase[junction]

            



        avg_reward = np.mean(list(total_rewards.values()))
        avg_loss = np.mean(total_losses) if total_losses else 0.0

        episode_rewards.append(avg_reward)
        episode_losses.append(avg_loss)

        current_epsilon = agents['J1'].epsilon

        # temporary change to see difference
        if episode % 1 == 0:
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
        'reward_fn': reward_fn,
        'scenario': scenario,
        'episodes': len(episode_rewards),
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
