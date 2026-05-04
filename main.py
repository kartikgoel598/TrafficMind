from dotenv import load_dotenv
import os, sys

load_dotenv()
# WHEN SETTING ENV, BE CAREFUL OF NESTED FOLDER
# in .env it should be something like Sumo_Home=D:\Users\Downloads\sumo-win64-1.26.0\sumo-1.26.0
SUMO_HOME = os.getenv('Sumo_Home')

if not SUMO_HOME:
    raise EnvironmentError("Sumo_Home not set in .env file!")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = os.path.join("sumo", "configs", "basic.sumocfg")
EPISODES      = 100       # how many full simulations to run
BATCH_SIZE    = 64        # how many experiences to learn from at once
SAVE_EVERY    = 10        # save model weights every N episodes
MODEL_DIR     = "models"  # folder to save trained weights
USE_GUI       = False     # set True to watch the simulation window, no need during training
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

env = SumoEnvironment(
    config_path=CONFIG_PATH,
    reward_fn='local',   # options: 'local', 'cooperative', 'fairness'
    use_gui=USE_GUI
)

# One DQN agent + one replay buffer per intersection
agents = {
    junction: DQNAgent(state_size=env.state_size, action_size=env.action_size)
    for junction in env.intersections
}
buffers = {
    junction: ReplayBuffer(capacity=50000)
    for junction in env.intersections
}

print(f"Training on {len(env.intersections)} intersections: {env.intersections}")
print(f"Running for {EPISODES} episodes...\n")

for episode in range(1, EPISODES + 1):

    states = env.reset()
    real_state_size = states[env.intersections[0]].shape[0]
    print(f"Detected state size: {real_state_size}")

    agents = {
    junction: DQNAgent(state_size=real_state_size, action_size=env.action_size)
    for junction in env.intersections
    }

    buffers = {
        junction: ReplayBuffer(capacity=50000)
        for junction in env.intersections
    }

    done   = False
    total_rewards = {j: 0.0 for j in env.intersections}
    total_loss    = {j: 0.0 for j in env.intersections}
    loss_count    = {j: 0   for j in env.intersections}

    while not done:

        # Each agent picks an action for its intersection
        actions = {
            junction: agents[junction].select_action(states[junction])
            for junction in env.intersections
        }

        next_states, rewards, done = env.step(actions)

        # Store experience + learn
        for junction in env.intersections:
            buffers[junction].push(
                states[junction],
                actions[junction],
                rewards[junction],
                next_states[junction],
                float(done)
            )
            loss = agents[junction].train_step(buffers[junction], BATCH_SIZE)
            if loss is not None:
                total_loss[junction]  += loss
                loss_count[junction]  += 1

            total_rewards[junction] += rewards[junction]

        states = next_states

    # Decay epsilon after each episode
    for junction in env.intersections:
        agents[junction].decay_epsilon()

    # ── Logging ───────────────────────────────────────────────────────────────
    avg_reward = sum(total_rewards.values()) / len(env.intersections)
    avg_loss   = {
        j: (total_loss[j] / loss_count[j]) if loss_count[j] > 0 else 0.0
        for j in env.intersections
    }
    epsilon = agents[env.intersections[0]].epsilon

    print(f"Episode {episode:>3}/{EPISODES} | "
          f"Avg Reward: {avg_reward:>8.2f} | "
          f"Epsilon: {epsilon:.3f} | "
          f"Avg Loss: { {j: f'{v:.4f}' for j, v in avg_loss.items()} }")

    # ── Save weights ──────────────────────────────────────────────────────────
    if episode % SAVE_EVERY == 0:
        for junction in env.intersections:
            path = os.path.join(MODEL_DIR, f"{junction}_ep{episode}.pth")
            agents[junction].save(path)
        print(f"  → Models saved at episode {episode}")

env.close()
print("\nTraining finished.")