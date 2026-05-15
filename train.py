import os
from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer

def train(config_path, episodes, batch_size, save_every, model_dir, use_gui, reward_fn):
    os.makedirs(model_dir, exist_ok=True)

    env = SumoEnvironment(
        config_path=config_path,
        reward_fn=reward_fn,
        use_gui=use_gui
    )

    # Get real state size from first reset (removes the duplicate agent bug)
    states = env.reset()
    real_state_size = states[env.intersections[0]].shape[0]
    print(f"Detected state size: {real_state_size}")

    # One DQN agent + one replay buffer per intersection
    agents = {
        junction: DQNAgent(state_size=real_state_size, action_size=env.action_size)
        for junction in env.intersections
    }
    buffers = {
        junction: ReplayBuffer(capacity=50000)
        for junction in env.intersections
    }

    print(f"Training on {len(env.intersections)} intersections: {env.intersections}")
    print(f"Running for {episodes} episodes...\n")

    for episode in range(1, episodes + 1):
        states = env.reset()
        done = False
        total_rewards = {j: 0.0 for j in env.intersections}
        total_loss    = {j: 0.0 for j in env.intersections}
        loss_count    = {j: 0   for j in env.intersections}

        while not done:
            actions = {
                junction: agents[junction].select_action(states[junction])
                for junction in env.intersections
            }

            next_states, rewards, done = env.step(actions)

            for junction in env.intersections:
                buffers[junction].push(
                    states[junction],
                    actions[junction],
                    rewards[junction],
                    next_states[junction],
                    float(done)
                )
                loss = agents[junction].train_step(buffers[junction], batch_size)
                if loss is not None:
                    total_loss[junction]  += loss
                    loss_count[junction]  += 1
                total_rewards[junction] += rewards[junction]

            states = next_states

        for junction in env.intersections:
            agents[junction].decay_epsilon()

        # ── Logging ───────────────────────────────────────────────────────────
        avg_reward = sum(total_rewards.values()) / len(env.intersections)
        avg_loss = {
            j: (total_loss[j] / loss_count[j]) if loss_count[j] > 0 else 0.0
            for j in env.intersections
        }
        epsilon = agents[env.intersections[0]].epsilon

        print(f"Episode {episode:>3}/{episodes} | "
              f"Avg Reward: {avg_reward:>8.2f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Avg Loss: { {j: f'{v:.4f}' for j, v in avg_loss.items()} }")

        # ── Save weights ──────────────────────────────────────────────────────
        if episode % save_every == 0:
            for junction in env.intersections:
                path = os.path.join(model_dir, f"{junction}_ep{episode}.pth")
                agents[junction].save(path)
            print(f"  → Models saved at episode {episode}")

    env.close()
    print("\nTraining finished.")