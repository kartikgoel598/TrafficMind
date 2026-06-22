# TrafficMind

## Overview

TrafficMind is a deep reinforcement learning traffic signal control system built on **SUMO** and **TraCI**. Four intersections on a grid network (`J1`, `J2`, `J4`, `J5`) are each controlled by an independent **Double DQN** agent. The platform studies how reward function design affects multi-agent learning, traffic efficiency, and fairness.

**Research objective:** Isolate the effect of reward shaping on RL traffic signal control by keeping the network, algorithm, state representation, and training pipeline identical while varying only the reward function.

**Problem being solved:** Fixed-time and heuristic signal plans do not adapt to live congestion. Prior work shows RL can outperform static timing, but it is less clear how cooperative, fairness-aware, and pressure-based reward terms change learned behavior. TrafficMind provides a reproducible experimental stack—training, evaluation, baselines, metrics logging, and a live dashboard—to answer that question on a controlled 4-junction SUMO grid.

---

## Key Features

* **SUMO traffic simulation** : 1 s step length, 900 s episodes, `peak` (~1500 veh/hr) and `off_peak` (~600 veh/hr) demand
* **Double DQN** : main network selects actions, target network evaluates Q-values, Huber loss, gradient clipping, periodic target updates
* **Independent DQN (IQL)** : one agent per junction with separate replay buffers
* **Six reward functions** : `local`, `cooperative`, `fairness`, `pressure_local`, `pressure_cooperative`, `pressure_fairness`
* **Action masking** : switch actions blocked during yellow and minimum-green periods (in both action selection and bootstrapping)
* **Executed-action replay** : transitions store actions SUMO actually executed, not merely requested actions
* **Evaluation framework** : trained DQN vs. `fixed_time`, `random_legal`, `greedy_queue`, `webster_static` across multiple seeds
* **Live dashboard** : FastAPI + WebSocket side-by-side policy comparison with real-time KPI streaming
* **Experiment configuration** : `config.yaml` for RL hyperparameters, CLI flags for reward, scenario, episodes, batch size, warmup
* **Result visualization** : per-run plots (`utils/plotter.py`), batch comparison plots (`plot_result.py`), dashboard results pages
* **KPI logging** : waiting time, queue length, throughput, switch statistics logged every training episode
* **Webster static baseline** : simplified two-phase cycle timing computed from scenario flow rates

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TrafficMind Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    TraCI     ┌──────────────────┐                     │
│  │ SUMO Network │◄────────────►│  SumoEnvironment │                     │
│  │ (grid.net)   │              │  state / reward  │                     │
│  └──────────────┘              │  yellow FSM      │                     │
│         ▲                      └────────┬─────────┘                     │
│         │ routes                        │ 13-dim obs, scalar reward     │
│  peak / off_peak                        ▼                               │
│                              ┌──────────────────────┐                   │
│                              │ 4 × DQNAgent (IQL)   │                   │
│                              │ replay + train_step  │                   │
│                              └──────────┬───────────┘                   │
│                                         │                               │
│         ┌───────────────────────────────┼───────────────────────┐       │
│         ▼                               ▼                       ▼       │
│   main.py (train)              evaluate.py (benchmark)   dashboard/     │
│   outputs/ + results.json      evaluations/*.json        live JSON      │
│                                                                         │
│         └───────────────────────────────┬───────────────────────┘       │
│                                         ▼                               │
│                          plot_result.py → output_plots/                 │
└─────────────────────────────────────────────────────────────────────────┘
```

Each RL step maps to one `traci.simulationStep()`. All junction signal updates are applied simultaneously before the step. The environment enforces minimum green (15 s) and yellow (3 s) internally; illegal switch requests are recorded as action `0` in replay.

---

## Project Structure

```
TrafficMind/
├── agents/
│   ├── dqn.py                  # Double DQN agent (masking, target network, Huber loss)
│   └── replay_buffer.py        # Uniform experience replay (deque, capacity 100k)
├── config.yaml                 # RL hyperparameters (learning rate, epsilon schedule, gamma)
├── dashboard/
│   ├── server.py               # FastAPI backend (versus launcher, WebSocket, model discovery)
│   ├── runner.py               # Single-episode runner writing live KPI JSON
│   ├── static/style.css        # Dashboard styles
│   └── templates/              # home, comparison, results, training_graph pages
├── environment/
│   └── sumo_env.py             # SUMO TraCI wrapper (state, rewards, KPIs, yellow FSM)
├── evaluations/                # Evaluation JSON outputs (from evaluate.py --all)
├── evaluate.py                 # Benchmark trained DQN vs. baselines
├── evaluate_extension.py       # Heatmap of per-intersection waiting time and phase starvation time-series
├── main.py                     # Training entry point
├── output_plots/               # Batch plots from plot_result.py
├── outputs/                    # Training runs (models, results.json, results.csv)
├── plot_result.py              # Cross-experiment training + KPI comparison plots
├── requirements.txt            # Python dependencies
├── rewards/
│   ├── local.py                # Queue + wait penalty (local only)
│   ├── cooperative.py          # + neighbour queue penalty
│   ├── fairness.py             # + phase starvation (red-time) penalty
│   ├── pressure_local.py       # + wrong-direction pressure term
│   ├── pressure_cooperative.py # cooperative + pressure
│   └── pressure_fairness.py    # fairness + pressure
├── simulate.py                 # Run a trained checkpoint in SUMO GUI
├── statistical_analysis.py     # Conduct paired t-tests comparing DQN formulations against baselines 
├── sumo/
│   ├── configs/                # peak.sumocfg, off_peak.sumocfg (900 s, 1 s steps)
│   ├── net/grid.net.xml        # 4 signalised junctions
│   └── routes/                 # peak.rou.xml, off_peak.rou.xml
└── utils/
    ├── logger.py               # Per-episode CSV logger
    ├── plotter.py              # Single-run reward/loss curves
    ├── traffic_signal_utils.py # Green/red lane queue helpers (pressure rewards)
    └── webster_utils.py        # Webster cycle timing for static baseline
```

---

## Installation

### Requirements

| Category | Requirement |
|---|---|
| **Python** | 3.10+ (tested with packages in `requirements.txt`) |
| **SUMO** | 1.26.0 (`sumo`, `sumo-gui`, `traci`, `sumolib`) |
| **GPU** | Optional - PyTorch uses CUDA when available |

### External software - SUMO 1.26.0

1. Download from [SUMO installation docs](https://sumo.dlr.de/docs/Installing/index.html) (e.g. `sumo-win64-1.26.0.zip`).
2. Extract so that `bin/sumo.exe` (or `bin/sumo` on Linux/macOS) exists under your install root.
3. Create a `.env` file in the project root (same folder as `main.py`):

```
Sumo_Home=D:\path\to\sumo-1.26.0
```

The path must point to the folder containing `bin/`, not the `bin/` folder itself.

### Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Core packages** (from `requirements.txt`):

| Package | Role |
|---|---|
| `torch`, `torchvision`, `torchaudio` | Neural networks and training |
| `traci`, `sumolib` | SUMO Python API |
| `numpy`, `pandas` | Numerics and data handling |
| `matplotlib`, `seaborn` | Plotting |
| `PyYAML` | `config.yaml` loading |
| `python-dotenv` | `Sumo_Home` from `.env` |
| `fastapi`, `uvicorn[standard]` | Dashboard backend |

---

## Quick Start

All commands below assume the virtual environment is activated and the working directory is the project root (`TrafficMind/`).

### 1. Installation

Follow [Installation](#installation) above.

### 2. Configuration

Edit `config.yaml` for learning-rate and exploration settings. Set `Sumo_Home` in `.env`.

### 3. Training

Default run (local reward, peak scenario, 700 episodes):

```bash
python main.py
```

Explicit examples:

```bash
python main.py --reward cooperative --scenario off_peak --episodes 700
python main.py --reward pressure_fairness --scenario peak --episodes 700 --seed 42
python main.py --reward fairness --scenario peak --episodes 300 --gui
```

### 4. Evaluation

Single model directory:

```bash
python evaluate.py --models-dir outputs/local_peak --scenario peak --reward local
```

Evaluate all preconfigured reward/scenario combinations:

```bash
python evaluate.py --all
```

### 5. Dashboard

Generate batch plots first (optional but required for Results / Training Graph pages):

```bash
python plot_result.py
```

Start the dashboard (from the `dashboard/` directory):

```bash
cd dashboard
uvicorn server:app --port 8080
```

Open `http://localhost:8080` for the overview. Use `/comparison` for live side-by-side policy runs.


### 6. Heatmap waiting time and phase starvation

Plot heatmap of per-intersection waiting time across conditions and formulations and phase starvation time-series for fairness analysis

```bash
python evaluate_extension.py
```

### 7. Paired t-tests

Conduct paired t-tests comparing DQN formulations against baselines 

```bash
python statistical_analysis.py
```

### Other useful commands

Per-run training curves:

```bash
python utils/plotter.py outputs/local_peak_20260616_174501
```

Watch a trained checkpoint in SUMO GUI:

```bash
python simulate.py --checkpoint-dir outputs/local_peak --scenario peak --reward local
```

---

## Configuration

### `config.yaml`

Loaded at training start by `main.py`:

| Key | Default | Description |
|---|---|---|
| `learning_rate` | `0.0005` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.01` | Minimum exploration rate |
| `epsilon_decay` | `0.999988` | Per environment-step multiplicative decay |

### CLI hyperparameters (`main.py`)

| Argument | Options / type | Default | Description |
|---|---|---|---|
| `--reward` | `local`, `cooperative`, `fairness`, `pressure_local`, `pressure_cooperative`, `pressure_fairness` | `local` | Reward function |
| `--scenario` | `peak`, `off_peak` | `peak` | Traffic demand |
| `--episodes` | int | `700` | Training episodes |
| `--gui` | flag | off | Enable SUMO GUI (slow) |
| `--seed` | int | `42` | Python / NumPy / PyTorch seed |
| `--batch_size` | int | `128` | Replay batch size |
| `--replay_warmup` | int | `5000` | Min transitions per agent before gradient updates |
| `--randomize_episode_seed` | flag | off | Random SUMO seed per episode instead of `seed + episode` |

### Training settings (code defaults)

| Setting | Value |
|---|---|
| Target network update | Every 500 gradient steps |
| Replay buffer capacity | 100,000 per junction |
| Gradient update frequency | Every 4 environment steps (when warmup met) |
| Network architecture | MLP `13 → 128 → 128 → 2` |
| Loss | Huber (`SmoothL1Loss`) |
| Gradient clipping | `max_norm=1.0` |
| Checkpoint interval | Every 100 episodes + `*_final.pth` |

### Environment settings (`sumo_env.py`)

| Setting | Value |
|---|---|
| Junctions | `J1`, `J2`, `J4`, `J5` |
| Actions | `0` = keep green, `1` = switch phase |
| Min green | 15 simulation seconds |
| Yellow duration | 3 simulation seconds |
| Episode length | Up to 900 steps or until no vehicles expected |
| Step length | 1 s (from `.sumocfg`) |

---

## Reinforcement Learning Pipeline

### State representation

Each junction agent receives a **13-dimensional** `float32` vector:

| Index | Feature | Description |
|---|---|---|
| 0–7 | Lane traffic | 4 approaches × (`queue/50`, `wait/300`); wait capped at 300 s per lane |
| 8 | Phase | Logical phase (`0` = horizontal green family, `1` = vertical), normalised |
| 9 | Phase time | `min(phase_time / min_green_time, 1.0)` — values &lt; 1.0 mean switching is blocked |
| 10 | Yellow flag | `1.0` during yellow countdown, else `0.0` |
| 11–12 | Neighbour queues | Mean queue per neighbour junction, normalised by `/50` |

### Action space

| Action | Meaning |
|---|---|
| `0` | Keep current green phase |
| `1` | Initiate phase switch (yellow → opposite green) |

Switching is blocked while `_yellow_timer > 0` or `_phase_time < min_green_time`. The agent masks action `1` in `select_action` and during Double DQN bootstrapping when illegal.

SUMO signal indices per junction: `0` green horizontal, `1` yellow horizontal, `2` green vertical, `3` yellow vertical.

### Reward functions

See [Reward Functions](#reward-functions). All scalar rewards are divided by **100.0** before being returned to the agent.

### Training process

1. `env.reset()` starts SUMO with `--seed` and runs one warmup step.
2. Each junction selects an action (ε-greedy with masking).
3. `env.step()` applies all signal changes, advances SUMO, computes rewards and next states.
4. Transitions `(state, executed_action, reward, next_state, done)` are pushed to per-junction replay buffers.
5. Every 4 steps, if buffer length ≥ `replay_warmup`, each agent runs `train_step`.
6. ε is decayed globally each step.
7. Episode KPIs and rewards are logged, checkpoints saved every 100 episodes.

**Logged episode reward:** mean cumulative reward across the four junctions (training proxy, not identical to SUMO delay KPIs).

### Evaluation process

`evaluate.py` loads `agent_J*_final.pth` checkpoints (ε = 0), runs one episode per seed, and compares policies:

* `trained_dqn` : loaded checkpoints
* `fixed_time` : switch every 20 s of green
* `random_legal` : random keep/switch when legal
* `greedy_queue` : switch when red-side queue exceeds green-side queue
* `webster_static` : switch after Webster-computed green time for current phase

Results include per-seed KPIs and mean ± std aggregates printed as a comparison table.

---

## Reward Functions

All modes share a comparable base structure. Pressure variants add a term penalising switches that move green away from the heavier queue side. Step rewards are scaled by `/100`.

| Mode | Formula (conceptual) | Extra parameters | Purpose |
|---|---|---|---|
| `local` | `-(own_queue + α·own_wait)` | α = 0.5 | Optimise local intersection only |
| `cooperative` | `-(own_queue + α·own_wait + β·Σ neigh_queue)` | α = 0.5, β = 0.3 | Penalise neighbour congestion |
| `fairness` | cooperative + `λ·max_red_time` | λ = 0.1 | Reduce phase starvation via internal red-time counter |
| `pressure_local` | `-(base + β·wrong_action_pressure)` | β = 2.0 | Penalise keeping green on low-pressure side or switching away from high-pressure side |
| `pressure_cooperative` | cooperative base + `γ·wrong_action_pressure` | γ = 2.0 | Cooperative terms + pressure alignment |
| `pressure_fairness` | fairness base + `γ·wrong_action_pressure` | γ = 1.5 | Fairness terms + pressure alignment |

**Definitions:**

* `own_queue` / `neigh_queue` — sum of halting vehicles on approach lanes (`getLastStepHaltingNumber`)
* `own_wait` — sum of per-lane waiting times, each capped at 300 s
* `max_red_time` — `max(_red_time[junction]) / 300.0` where `_red_time` counts steps each logical phase has been red
* `wrong_action_pressure` — if action = keep: `max(red_queue - green_queue, 0)`; if action = switch: `max(green_queue - red_queue, 0)`; green/red queues from live signal state via `get_green_red_queues`

---

## Evaluation

### Metrics collected

| Metric | Description |
|---|---|
| `mean_waiting_time` | Mean `getWaitingTime` over vehicles with wait &gt; 0, averaged across steps |
| `total_waiting_time` | Sum of `lane.getWaitingTime()` over all 16 approach lanes per step, averaged across steps |
| `mean_queue_length` | Mean halting vehicles per lane per step |
| `max_lane_wait` | Running maximum lane waiting time in the episode |
| `throughput` | Cumulative `getArrivedNumber()` across steps |
| `switch_count_total` | Executed phase switches (action `1` actually applied) |
| `mean_reward_per_step` | Mean per-step reward (evaluation only) |
| `step_count` | Steps in the episode |

Training also logs per-episode switch diagnostics: `legal_switch_opportunities`, `switch_when_legal`, `keep_when_legal`, `illegal_switch_requests`, and per-junction phase change counts.

### Comparison process

1. Run `evaluate.py` with `--models-dir` or `--all`.
2. Each policy runs one episode per seed (default: `42, 123, 456, 789, 1000`).
3. KPIs are aggregated as mean ± std per policy.
4. A console comparison table is printed; JSON is written to `evaluations/`.

### Output files

| Path | Contents |
|---|---|
| `evaluations/<reward>_<scenario>_results.json` | Full per-policy, per-seed KPIs and aggregates |
| `evaluations/evaluation_results.json` | Default path for single-run evaluation (`--output`) |

---

## Dashboard

### Features

| Page | URL | Purpose |
|---|---|---|
| Overview | `/` | Project description, goals, reward list, tech stack |
| Comparison | `/comparison` | Side-by-side live simulation of two policies |
| Results | `/results` | KPI bar charts from `output_plots/` |
| Training Graph | `/training_graph` | Training reward/loss curves |

**Comparison page policies:** `trained_dqn`, `fixed_time`, `random_legal`, `greedy_queue`, `webster_static`.

**Live metrics streamed via WebSocket:** mean waiting time, queue length, total waiting time, max lane wait, throughput, switch count, simulation time.

### Usage

```bash
python plot_result.py          # generate plots first (from project root)
cd dashboard
uvicorn server:app --port 8080
```

Select two agents, scenario, reward function, and seed on the Comparison page, then launch. Two SUMO GUI instances run in parallel (one per slot).

### Data flow

```
Browser → POST /run → server.py spawns two runner.py processes
                              ↓
                    each writes live_a.json / live_b.json per step
                              ↓
Browser ← WebSocket /ws ← server polls JSON every 500 ms
```

`GET /models` lists `outputs/` folders containing `agent_*_final.pth` checkpoints.

---

## Results

### Training outputs (`outputs/<reward>_<scenario>_<timestamp>/`)

| File | Description |
|---|---|
| `agent_J1_final.pth` … `agent_J5_final.pth` | Final model weights per junction |
| `agent_J*_ep100.pth`, `ep200`, … | Periodic checkpoints |
| `results.json` | Episode rewards, losses, KPIs, hyperparameters |
| `results.csv` | Per-episode CSV log (reward, loss, ε, per-junction rewards) |

`results.json` fields include: `reward_fn`, `scenario`, `episodes`, `seed`, `batch_size`, `replay_warmup`, `episode_rewards`, `episode_rewards_per_step`, `episode_losses`, `episode_kpis`.

### Visualization outputs (`output_plots/`)

Generated by `plot_result.py`:

| File | Description |
|---|---|
| `rewards_original.png` | Training rewards for local / cooperative / fairness |
| `rewards_pressure.png` | Training rewards for pressure variants |
| `loss.png` | Loss curves (all reward functions, comparable) |
| `rewards_normalized.png`, `loss_normalized.png` | Normalised cross-condition comparisons |
| `kpi_<metric>_peak.png`, `kpi_<metric>_off_peak.png` | Grouped bar charts (DQN vs. baselines) |
| `original_rewards/`, `pressure_rewards/` | Subfolder KPI charts by reward family |

### Per-run plots (`utils/plotter.py`)

* `reward_curve.png` — raw + smoothed episode rewards
* `loss_curve.png` — raw + smoothed episode losses

---

## Current Limitations

* **Independent learners (IQL)** - agents do not share parameters or gradients; cooperative/fairness rewards add cross-junction terms but training remains decentralised.
* **Fixed network topology** - four 2-phase junctions on a single grid; not generalised to arbitrary networks.
* **Short episodes** - 900 s per episode although route files define 3600 s of demand.
* **Reward scale differs across formulations** — raw `episode_rewards` are not directly comparable across reward modes; use SUMO KPIs for cross-reward comparison.
* **Dashboard SUMO config paths** - `dashboard/runner.py` resolves SUMO configs relative to the `dashboard/` directory (`dashboard/sumo/...`), while configs live at project-root `sumo/`. Running the dashboard may require path alignment or launching from a corrected working directory.
* **`simulate.py` reward choices** - CLI only lists `local`, `cooperative`, `fairness`, `pressure_local` (not all six pressure variants).
* **No multi-agent coordination algorithm** - no QMIX, MADDPG, or centralized critic.
* **Uniform replay only** - no prioritized experience replay.
* **Evaluation depends on checkpoint layout** - expects `agent_J1_final.pth` … `agent_J5_final.pth` in the models directory.

---

## Future Improvements

* Fix dashboard runner paths to resolve SUMO configs from the project root.
* Add centralized or communication-based multi-agent training (e.g. parameter sharing, graph attention).
* Extend evaluation to longer horizons and additional demand patterns.
* Log standard SUMO output statistics (tripinfo, emission) for richer KPIs.
* Prioritized replay and learning-rate scheduling.
* Headless dashboard mode (no dual SUMO GUI requirement).
* Hyperparameter search / experiment tracking (e.g. Weights & Biases).
* Generalise to larger networks and variable junction counts.

---

## Technologies Used

| Technology | Use |
|---|---|
| [Eclipse SUMO](https://eclipse.dev/sumo/) 1.26 | Microscopic traffic simulation |
| TraCI | Real-time simulation control |
| PyTorch 2.x | Double DQN implementation |
| NumPy | State vectors and KPI aggregation |
| Pandas | Data handling (dependency) |
| Matplotlib / Seaborn | Training and evaluation plots |
| FastAPI | Dashboard REST + WebSocket API |
| Uvicorn | ASGI server |
| PyYAML | Hyperparameter config |
| python-dotenv | Environment variable loading |

---

## License

Distributed under the MIT License.

---

## Troubleshooting

**`Sumo_Home` / SUMO not found**  
Ensure `.env` exists in the project root and points to the SUMO install containing `bin/sumo.exe`.

**`FileNotFoundError` when starting simulation**  
Verify SUMO configs from the project root:

```bash
"<your_sumo_path>/bin/sumo.exe" -c "sumo/configs/peak.sumocfg"
```

**`Connection closed by SUMO`**  
Usually a bad network, route, or config path. Run SUMO directly to see the error.

**Model fails to load**  
Checkpoints store `state_size` and `action_size`. Models trained with a different state dimension (e.g. before the yellow-flag feature) are incompatible with the current 13-dimensional state.

**Training reward flat but KPIs improve**  
`episode_rewards` is a shaped training signal, not mean waiting time. Inspect `episode_kpis` in `results.json` or run `evaluate.py`.

**Avoid `--gui` for long training**  
SUMO GUI significantly slows simulation.
