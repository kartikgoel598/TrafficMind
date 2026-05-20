# TrafficMind

A deep reinforcement learning–based traffic signal control system using SUMO. Compares three reward functions — local intersection performance, network-wide coordination, and fairness between directions — to understand how reward design shapes learning and provide evidence-based guidelines for better traffic control.

---

## Prerequisites

- Python 3.10+
- SUMO 1.26.0 (see Step 1)

---

## Setup

### Step 1 — Download SUMO

1. Go to https://sumo.dlr.de/docs/Installing/index.html
2. Download **"64-bit zip: sumo-win64-1.26.0.zip"**
3. Extract the zip somewhere on your machine, for example:
   ```
   D:\sumo\sumo-win64-1.26.0
   ```
4. Confirm the following path exists after extraction:
   ```
   D:\sumo\sumo-win64-1.26.0\sumo-1.26.0\bin\sumo.exe
   ```

---

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Create a `.env` file

In the project root (same folder as `main.py`), create a file called `.env` and add the path to your extracted SUMO folder:

```
Sumo_Home=D:\sumo\sumo-win64-1.26.0\sumo-1.26.0
```

> ⚠️ Make sure the path points to the folder that contains the `bin\` subfolder — not the `bin\` folder itself.

**Example paths depending on where you extracted SUMO:**

| Extracted to | `.env` value |
|---|---|
| `D:\sumo\sumo-win64-1.26.0` | `Sumo_Home=D:\sumo\sumo-win64-1.26.0\sumo-1.26.0` |
| `C:\Users\yourname\Downloads\sumo-win64-1.26.0` | `Sumo_Home=C:\Users\yourname\Downloads\sumo-win64-1.26.0\sumo-1.26.0` |

---

### Step 5 — Run the simulation

Basic run with defaults (local reward, peak scenario, 500 episodes):

```bash
python main.py
```

With options:

```bash
python main.py --reward cooperative --scenario off_peak --episodes 300 --gui --seed 42
```

Resume training from a checkpoint:

```bash
python main.py --resume outputs/local_peak_20250101_120000 --start_episode 201 --episodes 500
```

---

## CLI Arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--reward` | `local`, `cooperative`, `fairness` | `local` | Reward function to use |
| `--scenario` | `peak`, `off_peak` | `peak` | Traffic scenario |
| `--episodes` | any int | `500` | Number of training episodes |
| `--gui` | flag | off | Enable SUMO visual GUI |
| `--seed` | any int | `42` | Random seed for reproducibility |
| `--resume` | path | `None` | Path to checkpoint folder to resume from |
| `--start_episode` | any int | `1` | Episode number to resume from |

---

## Reward Functions

All three reward functions share the same base signal (`own_queue + α·own_wait`) so comparisons are fair — only the additional penalty terms differ.

| Mode | Formula | Purpose |
|---|---|---|
| `local` | `-(own_queue + α·own_wait)` | Optimise own intersection only |
| `cooperative` | `-(own_queue + α·own_wait + β·neigh_queue)` | Coordinate with neighbouring junctions |
| `fairness` | `-(own_queue + α·own_wait + β·neigh_queue + λ·starvation)` | Prevent any direction from being starved |

Parameters: `α=0.5`, `β=0.3`, `λ=0.1`. All outputs normalised by `/100.0`.

---

## State Space (12 features per agent)

Each agent observes a 12-dimensional state vector:

| Features | Count | Description |
|---|---|---|
| Queue per lane | 4 | Normalised by `/50.0` |
| Wait per lane | 4 | Per-vehicle average, normalised by `/300.0` |
| Current phase | 1 | Read directly from SUMO via TraCI |
| Phase time | 1 | `phase_time / min_green_time`, capped at `1.0` — tells agent if switching is legal |
| Neighbour queue | 2 | One per neighbour junction, normalised |

---

## Project Structure

```
TrafficMind/
├── agents/
│   ├── dqn.py               # DQN agent (QNetwork + training loop)
│   └── replay_buffer.py     # Experience replay buffer
├── environment/
│   └── sumo_env.py          # SUMO TraCI environment wrapper
├── rewards/
│   ├── local.py             # Local reward function
│   ├── cooperative.py       # Cooperative reward function
│   └── fairness.py          # Fairness reward function
├── sumo/
│   ├── configs/
│   │   ├── peak.sumocfg     # Peak hour config
│   │   └── off_peak.sumocfg # Off-peak config
│   ├── net/
│   │   └── grid.net.xml     # Road network (4 junctions: J1, J2, J4, J5)
│   └── routes/
│       ├── peak.rou.xml     # Peak hour vehicle routes
│       └── off_peak.rou.xml # Off-peak vehicle routes
├── outputs/                 # Auto-created; stores models + results per run
│   └── <reward>_<scenario>_<timestamp>/
│       ├── agent_J1_final.pth
│       ├── agent_J2_final.pth
│       ├── agent_J4_final.pth
│       ├── agent_J5_final.pth
│       └── results.json
├── .env                     # Your local SUMO path (not committed to git)
├── main.py                  # Entry point — training loop
├── requirements.txt
└── README.md
```

---

## Outputs

Each training run creates a timestamped folder under `outputs/`:

```
outputs/local_peak_20250101_120000/
├── agent_J1_final.pth      # Final model weights
├── agent_J2_final.pth
├── agent_J4_final.pth
├── agent_J5_final.pth
├── agent_J1_ep100.pth      # Checkpoint every 100 episodes
├── ...
└── results.json            # Episode rewards, losses, config
```

`results.json` contains:
```json
{
  "reward_fn": "local",
  "scenario": "peak",
  "episodes": 500,
  "seed": 42,
  "episode_rewards": [...],
  "episode_losses": [...]
}
```

---

## Troubleshooting

**`Sumo_Home not set in .env file!`**
→ Make sure your `.env` file exists in the project root and is spelled correctly.

**`FileNotFoundError` when starting simulation**
→ Your `Sumo_Home` path is wrong. Double check that `bin\sumo.exe` exists inside the path you provided.

**`Connection closed by SUMO`**
→ SUMO started but crashed. Run this to see the actual error:
```bash
"<your_sumo_path>\bin\sumo.exe" -c "sumo\configs\peak.sumocfg"
```

**Model fails to load on `--resume`**
→ If you changed `state_size` (e.g. from 11 to 12 after adding `phase_time`), old `.pth` files are incompatible. You need to retrain from scratch.

---

## Notes

- The `.env` file is intentionally excluded from git (listed in `.gitignore`) since each team member's SUMO path will be different.
- The simulation runs for a maximum of 900 steps per episode, or until all vehicles have cleared.
- Junction processing order is randomised each step to remove ordering bias from training.
- Models are saved every 100 episodes and at the end of training.
