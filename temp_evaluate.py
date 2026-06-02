# simulate.py
import os, sys, random
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.getenv('Sumo_Home'), "tools"))

from environment.sumo_env import SumoEnvironment
from agents.dqn import DQNAgent

CHECKPOINT_DIR = 'outputs/local_peak_20260602_142746'  
intersections = ['J1', 'J2', 'J4', 'J5']

env = SumoEnvironment(
    config_path='sumo/configs/peak.sumocfg',
    reward_fn='local',
    use_gui=True,   
    seed=42
)

agents = {}
for junction in intersections:
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
    )
    agent.load(os.path.join(CHECKPOINT_DIR, f'agent_{junction}_final.pth'))
    agent.epsilon = 0.0  # pure exploitation, no random actions
    agents[junction] = agent

states = env.reset()
done = False
switch_counts = {j: 0 for j in intersections}
step = 0

while not done:
    actions = {}
    for junction in intersections:
        actions[junction] = agents[junction].select_action(states[junction])
        if actions[junction] == 1:
            switch_counts[junction] += 1

    next_states, rewards, done, executed_actions = env.step(actions)
    states = next_states
    step += 1

print(f"\nSteps: {step}")
print(f"Switch counts: {switch_counts}")
total = sum(switch_counts.values())
print(f"Total switches: {total} | Switch rate: {total/(step*4):.2%}")

env.close()