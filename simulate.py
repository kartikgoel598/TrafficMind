# simulate.py — run a trained checkpoint in SUMO GUI with switch stats
import os
import sys
import argparse

from dotenv import load_dotenv

load_dotenv()

sumo_home = os.getenv("Sumo_Home")
if not sumo_home:
    raise EnvironmentError(
        "Sumo_Home is not set. Add it to your .env file (see README)."
    )
sys.path.append(os.path.join(sumo_home, "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent

INTERSECTIONS = ["J1", "J2", "J4", "J5"]


def parse_args():
    """Parse command line arguments for simulation."""
    parser = argparse.ArgumentParser(description="Run a trained checkpoint in SUMO")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/local_peak_final",
        help="Directory containing agent_J*_final.pth files",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="peak",
        choices=["peak", "off_peak"],
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="local",
        choices=["local", "cooperative", "fairness", "pressure_local"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--no-gui", action="store_false", dest="gui")
    return parser.parse_args()


def get_config_path(scenario: str) -> str:
    """Get the path to the SUMO configuration file for a given scenario."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = {
        "peak": os.path.join(base_dir, "sumo", "configs", "peak.sumocfg"),
        "off_peak": os.path.join(base_dir, "sumo", "configs", "off_peak.sumocfg"),
    }
    return configs[scenario]


def main():
    """Main entry point for running a trained checkpoint in SUMO."""
    args = parse_args()

    env = SumoEnvironment(
        config_path=get_config_path(args.scenario),
        reward_fn=args.reward,
        use_gui=args.gui,
        seed=args.seed,
    )

    agents = {}
    for junction in INTERSECTIONS:
        model_path = os.path.join(args.checkpoint_dir, f"agent_{junction}_final.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
        )
        agent.load(model_path)
        agent.epsilon = 0.0
        agents[junction] = agent

    states = env.reset()
    done = False
    switch_counts = {j: 0 for j in INTERSECTIONS}
    step = 0

    while not done:
        actions = {
            j: agents[j].select_action(states[j]) for j in INTERSECTIONS
        }
        next_states, _rewards, done, executed_actions = env.step(actions)
        for junction in INTERSECTIONS:
            switch_counts[junction] += int(executed_actions[junction] == 1)
        states = next_states
        step += 1

    print(f"\nSteps: {step}")
    print(f"Switch counts (executed): {switch_counts}")
    total = sum(switch_counts.values())
    print(f"Total switches: {total} | Switch rate: {total / (step * 4):.2%}")

    env.close()


if __name__ == "__main__":
    main()
