# 
from dotenv import load_dotenv
import os, sys

load_dotenv()
# WHEN SETTING ENV, BE CAREFUL OF NESTED FOLDER
# in .env it should be something like Sumo_Home=D:\Users\Downloads\sumo-win64-1.26.0\sumo-1.26.0
SUMO_HOME = os.getenv('Sumo_Home')

if not SUMO_HOME:
    raise EnvironmentError("Sumo_Home not set in .env file!")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

# Now import the environment
from environment.sumo_env import SumoEnvironment

# Point to one of your config files
CONFIG_PATH = os.path.join("sumo", "configs", "basic.sumocfg")

env = SumoEnvironment(
    config_path=CONFIG_PATH,
    reward_fn='local',
    use_gui=True  # set True if you want the GUI window
)

# Run a simple test loop
state = env.reset()
print("Simulation started! Initial state keys:", list(state.keys()))

done = False
while not done:
    # Random actions for now: 0 = keep phase, 1 = switch phase
    actions = {junction: 0 for junction in env.intersections}
    state, rewards, done = env.step(actions)

env.close()
print("Simulation finished.")


"""
from dotenv import load_dotenv
import os, sys

load_dotenv()
SUMO_HOME = os.getenv('Sumo_Home')

# Safety check
if not SUMO_HOME:
    raise EnvironmentError("Sumo_Home not set in .env file!")

# Add tools to path so traci/sumolib imports work
sys.path.append(os.path.join(SUMO_HOME, "tools"))

print(SUMO_HOME)  # should now print your path
"""
