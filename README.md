# TrafficMind

A deep reinforcement learning traffic signal control system built on **SUMO** and **TraCI**. Four intersections (J1, J2, J4, J5) on a grid network are each controlled by an independent **DQN** agent. The project compares three reward designs — **local**, **cooperative**, and **fairness** — to study how reward shaping affects multi-agent learning.

---

## System overview

| Component | Description |
|-----------|-------------|
| Simulation | SUMO 1.26, 1 s step length, up to **900 steps** per episode |
| Agents | One DQN per junction (`J1`, `J2`, `J4`, `J5`) |
| Actions | `0` = keep current green, `1` = switch phase (after min green + yellow) |
| Algorithm | DQN with experience replay, ε-greedy exploration, target network |
| Scenarios | `peak` (~1500 veh/hr) and `off_peak` (~600 veh/hr) |

**Environment behaviour (current):**

- One TraCI `simulationStep()` per RL step (fixed decision interval).
- All signal updates are applied **simultaneously** before each step.
- **Yellow** is handled by an internal countdown (`yellow_duration = 3`); switching is blocked while yellow is active.
- **Minimum green** (`min_green_time = 15` steps) is enforced in the environment; illegal switch requests are stored as action `0` in replay.
- `reset()` runs one warmup simulation step before the first observation.

**Agent behaviour:**

- **Action masking** in both `select_action` and `train_step`: switch is disallowed when `phase_time < 1.0` (normalised) or `is_yellow == 1.0`.
- Replay stores **executed** actions from SUMO, not merely intended actions.

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

> Make sure the path points to the folder that contains the `bin\` subfolder — not the `bin\` folder itself.

| Extracted to | `.env` value |
|---|---|
| `D:\sumo\sumo-win64-1.26.0` | `Sumo_Home=D:\sumo\sumo-win64-1.26.0\sumo-1.26.0` |
| `C:\Users\yourname\Downloads\sumo-win64-1.26.0` | `Sumo_Home=C:\Users\yourname\Downloads\sumo-win64-1.26.0\sumo-1.26.0` |

---

### Step 5 — Run training

Used run by developer
```bash
python main.py --reward local --scenario peak --episodes 300
python main.py --reward cooperative --scenario off_peak --episodes 200
python main.py --reward fairness --scenario peak --episodes 300

python main.py --reward local --scenario peak --episodes 700 --resume outputs/local_peak_TIMESTAMP --start_episode 301
python main.py --reward cooperative --scenario peak --episodes 500 --resume outputs/cooperative_peak_20260522_135204 --start_episode 301
python main.py --reward cooperative --scenario off_peak --episodes 300 --resume outputs/cooperative_off_peak_20260525_160322 --start_episode 201

python main.py --reward cooperative --scenario peak --episodes 700 --resume outputs/cooperative_peak_20260522_135204 --start_episode 501
python main.py --reward fairness --scenario peak --episodes 700 --resume outputs/fairness_peak_TIMESTAMP --start_episode 301
```

Basic run (local reward, peak scenario, 500 episodes):

```bash
python main.py
```

With options:

```bash
python main.py --reward cooperative --scenario off_peak --episodes 300 --gui --seed 42
```

NOTE: RESUME DOESNT WORK AS OF CURRENTLY
Resume training to episodes(loads `agent_<junction>_final.pth` from the output folder):

```bash
python main.py --resume outputs/local_peak_20260602_142746 --episodes 800
```

Plot results from a completed run:

```bash
python utils/plotter.py outputs/local_peak_20250101_120000
```

---

## CLI arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--reward` | `local`, `cooperative`, `fairness` | `local` | Reward function |
| `--scenario` | `peak`, `off_peak` | `peak` | Traffic demand file |
| `--episodes` | any int | `500` | Number of training episodes |
| `--gui` | flag | off | Enable SUMO GUI (much slower) |
| `--seed` | any int | `42` | Seed for Python / NumPy / PyTorch |
| `--resume` | path | `None` | Output directory to resume from |
| `--start_episode` | any int | `1` | Episode index when resuming |

---

## Reward functions

All three modes share the same base penalty on the controlled intersection. Only the extra terms differ, so ablations are comparable. Each step reward is divided by **100.0**.

| Mode | Formula | Purpose |
|---|---|---|
| `local` | `-(own_queue + α·own_wait)` | Optimise the local intersection only |
| `cooperative` | `-(own_queue + α·own_wait + β·Σ neigh_queue)` | Penalise congestion at neighbouring junctions |
| `fairness` | `-(own_queue + α·own_wait + β·Σ neigh_queue + λ·max_lane_wait)` | Add pressure to reduce the worst lane waiting time |

**Parameters:** `α = 0.5`, `β = 0.3`, `λ = 0.1`

- `own_queue` / `neigh_queue`: sum of halting vehicles on approach lanes.
- `own_wait`: sum of per-lane waiting times, capped at **300 s** per lane before summing.
- **Fairness** `max_lane_wait`: maximum `getWaitingTime` over the junction’s approach lanes, divided by **300.0** (normalised starvation signal).

---

## State space (13 features per agent)

Each junction agent receives a **13-dimensional** `float32` vector:

| Index | Feature | Description |
|---|---|---|
| 0–7 | Lane traffic | 4 approaches × (`queue/50`, `avg_wait/300`); avg wait = lane waiting time ÷ max(1, queue) |
| 8 | Phase | Logical phase from SUMO (`0` = horizontal green family, `1` = vertical), normalised |
| 9 | Phase time | `min(phase_time / min_green_time, 1.0)` — values **&lt; 1.0** mean switching is still blocked |
| 10 | Yellow flag | `1.0` during yellow countdown, else `0.0` |
| 11–12 | Neighbour queues | Mean queue per neighbour junction, normalised by `/50` |

Phase and yellow are read from TraCI / internal timers so observations stay consistent during transitions.

---

## Actions and signal timing

| Action | Meaning |
|---|---|
| `0` | Keep current green phase |
| `1` | Initiate phase switch (yellow → opposite green) |

**Constraints:**

- **Min green:** 15 simulation seconds before a new switch is accepted.
- **Yellow:** 3 simulation seconds; no new switch while `_yellow_timer > 0`.
- Agents and the learner **mask** action `1` when switching is illegal (see `agents/dqn.py`).

SUMO signal indices per junction: `0` green horizontal, `1` yellow horizontal, `2` green vertical, `3` yellow vertical.

---

## Training hyperparameters (defaults in `main.py`)

| Hyperparameter | Value |
|---|---|
| Learning rate | `0.0005` |
| Discount γ | `0.99` |
| ε start / min / decay | `1.0` / `0.01` / `0.990` (per episode) |
| Target network update | Every **50** gradient steps |
| Replay buffer size | `100,000` per junction |
| Batch size | `128` |
| Network | MLP `13 → 128 → 128 → 2` |

**Seeds:**

- `--seed` fixes Python, NumPy, and PyTorch RNGs at the start of training.
- Each episode sets `env.seed` to a **random** integer in `0–9999` for SUMO traffic variation across episodes.

**Logged episode reward:** mean of the **cumulative step rewards** over the four junctions (not normalised per step). Use this for training curves; for research comparisons you should also log SUMO-level KPIs (mean delay, throughput, etc.) separately.

---

## Project structure

```
TrafficMind/
├── agents/
│   ├── dqn.py               # DQN agent (masking, target network, training)
│   └── replay_buffer.py     # Uniform experience replay
├── environment/
│   └── sumo_env.py          # SUMO TraCI wrapper (yellow FSM, rewards, state)
├── rewards/
│   ├── local.py
│   ├── cooperative.py
│   └── fairness.py
├── sumo/
│   ├── configs/
│   │   ├── peak.sumocfg
│   │   └── off_peak.sumocfg
│   ├── net/
│   │   └── grid.net.xml     # 4 signalised junctions: J1, J2, J4, J5
│   └── routes/
│       ├── peak.rou.xml
│       └── off_peak.rou.xml
├── utils/
│   ├── logger.py            # CSV episode logger (optional integration)
│   └── plotter.py           # Reward / loss plots from results.json
├── outputs/                 # Created per run
│   └── <reward>_<scenario>_<timestamp>/
├── main.py                  # Training entry point
├── requirements.txt
└── README.md
```

---

## Outputs

Each run writes a timestamped folder under `outputs/`:

```
outputs/local_peak_20250101_120000/
├── agent_J1_final.pth
├── agent_J2_final.pth
├── agent_J4_final.pth
├── agent_J5_final.pth
├── agent_J1_ep100.pth       # Checkpoints every 100 episodes
├── ...
└── results.json
```

**Checkpoint naming:** `agent_<JUNCTION>_ep<N>.pth` and `agent_<JUNCTION>_final.pth` (e.g. `agent_J1_final.pth`). Resume expects the `_final` files unless you copy/rename episode checkpoints manually.

`results.json` example:

```json
{
  "reward_fn": "local",
  "scenario": "peak",
  "episodes": 500,
  "seed": 42,
  "episode_rewards": [],
  "episode_losses": []
}
```

---

## Troubleshooting

**`Sumo_Home` / SUMO not found**  
Ensure `.env` exists in the project root and points to the SUMO install that contains `bin/sumo.exe`.

**`FileNotFoundError` when starting simulation**  
Run SUMO directly to see the error:

```bash
"<your_sumo_path>/bin/sumo.exe" -c "sumo/configs/peak.sumocfg"
```

**`Connection closed by SUMO`**  
Usually a bad network, route, or config path. Confirm configs are run from the project root (relative paths in `.sumocfg`).

**Model fails to load on `--resume`**  
Checkpoints are tied to **state size**. Models trained with `state_size=12` or earlier are **not** compatible with the current **13-dimensional** state (yellow flag). Retrain from scratch after architecture changes.

**Resume finds no weights**  
Resume looks for `agent_J1_final.pth`, not `agent_J1_ep100.pth`. Use final weights or copy an episode checkpoint to the `_final` name.

**Training looks flat but SUMO improves**  
`episode_rewards` in `results.json` is a training proxy, not mean waiting time or throughput. Inspect SUMO statistics or add KPI logging for evaluation.

---

## Notes

- `.env` is gitignored; each machine sets its own `Sumo_Home`.
- Episodes end at **900 simulation steps** or when SUMO reports no further expected vehicles.
- Route files define demand for 3600 s; only the first **900 s** is simulated per episode.
- Avoid `--gui` for long training runs.
- Independent DQN (IQL) treats other junctions as part of the environment; cooperative/fairness rewards add cross-junction terms but do not use centralized training.
