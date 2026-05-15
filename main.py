from dotenv import load_dotenv
import os, sys

load_dotenv()

SUMO_HOME = os.getenv('Sumo_Home')
if not SUMO_HOME:
    raise EnvironmentError("Sumo_Home not set in .env file!")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

from train import train

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join("sumo", "configs", "peak.sumocfg")  # changed from basic
EPISODES    = 100
BATCH_SIZE  = 64
SAVE_EVERY  = 10
MODEL_DIR   = "models"
USE_GUI     = False  # changed to False, no need GUI during training
REWARD_FN   = 'local'  # options: 'local', 'cooperative', 'fairness'
# ──────────────────────────────────────────────────────────────────────────────

train(
    config_path=CONFIG_PATH,
    episodes=EPISODES,
    batch_size=BATCH_SIZE,
    save_every=SAVE_EVERY,
    model_dir=MODEL_DIR,
    use_gui=USE_GUI,
    reward_fn=REWARD_FN
)