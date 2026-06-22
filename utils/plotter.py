import matplotlib.pyplot as plt
import matplotlib
import json
import os

# Non-interactive backend — server pe bhi kaam kare
matplotlib.use('Agg')

def plot_results(output_dir):
    """Generate reward and loss plots from results.json."""

    results_path = os.path.join(output_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f"results.json nahi mila: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    episode_rewards = results['episode_rewards']
    episode_losses  = results['episode_losses']
    episodes        = list(range(1, len(episode_rewards) + 1))

    # ---- Plot 1 — Reward Convergence ----
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards,
             alpha=0.4, color='blue', label='Raw Reward')

    # Smoothing — moving average (window=20)
    # Kyun: raw reward noisy hota hai — trend dekhna mushkil
    # Moving average se trend clear hota hai
    window = min(20, len(episode_rewards) // 5)
    if window > 1:
        smoothed = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window)
            smoothed.append(
                sum(episode_rewards[start:i+1]) / (i - start + 1)
            )
        plt.plot(episodes, smoothed,
                 color='blue', linewidth=2, label=f'Smoothed (window={window})')

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f"Reward Convergence — {results.get('reward_fn', '')} "
              f"({results.get('scenario', '')})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    reward_path = os.path.join(output_dir, 'reward_curve.png')
    plt.savefig(reward_path, dpi=150)
    plt.close()
    print(f"Reward curve saved: {reward_path}")

    # ---- Plot 2 — Loss Curve ----
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_losses,
             alpha=0.4, color='red', label='Raw Loss')

    if window > 1:
        smoothed_loss = []
        for i in range(len(episode_losses)):
            start = max(0, i - window)
            smoothed_loss.append(
                sum(episode_losses[start:i+1]) / (i - start + 1)
            )
        plt.plot(episodes, smoothed_loss,
                 color='red', linewidth=2, label=f'Smoothed (window={window})')

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(f"Loss Curve — {results.get('reward_fn', '')} "
              f"({results.get('scenario', '')})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {loss_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        plot_results(sys.argv[1])
    else:
        print("Usage: python utils/plotter.py outputs\\1st_result")