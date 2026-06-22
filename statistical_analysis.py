import json
import os
import numpy as np
from scipy import stats

EVALUATIONS_DIR = "evaluations"
SEEDS = ["42", "123", "456", "789", "1000"]
BASELINES = ["fixed_time", "random_legal", "greedy_queue", "webster_static"]
KPI = "mean_waiting_time"  # or swap for other KPIs

def load_per_seed(json_path: str, policy: str) -> list:
    """Extract per-seed KPI values in seed order."""
    with open(json_path, "r") as f:
        data = json.load(f)
    per_seed = data["policies"][policy]["per_seed"]
    return [per_seed[s][KPI] for s in SEEDS]

def run_paired_ttests(json_path: str, model_name: str):
    print(f"\n{'='*80}")
    print(f"  {model_name}")
    print(f"{'='*80}")

    dqn_scores = load_per_seed(json_path, "trained_dqn")

    print(
        f"{'Baseline':<18} | {'DQN':>8} | {'Base':>8} | "
        f"{'Δ':>8} | {'t':>8} | {'p':>8} | Sig"
    )
    print("-" * 80)
    for baseline in BASELINES:
        baseline_scores = load_per_seed(json_path, baseline)

        # paired t-test: same seed = same conditions
        t_stat, p_value = stats.ttest_rel(dqn_scores, baseline_scores)

        dqn_mean = np.mean(dqn_scores)
        baseline_mean = np.mean(baseline_scores)
        diff = baseline_mean - dqn_mean  # positive = DQN is better

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(
            f"{baseline:<18} | "
            f"{dqn_mean:>8.3f} | "
            f"{baseline_mean:>8.3f} | "
            f"{diff:>+8.3f} | "
            f"{t_stat:>+8.3f} | "
            f"{p_value:>8.4f} | "
            f"{sig}"
        )

def main():
    for filename in sorted(os.listdir(EVALUATIONS_DIR)):
        if not filename.endswith("_results.json"):
            continue
        path = os.path.join(EVALUATIONS_DIR, filename)
        model_name = filename.replace("_results.json", "")
        run_paired_ttests(path, model_name)

if __name__ == "__main__":
    main()