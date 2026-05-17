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

```bash
python main.py
```

To run with the **visual GUI** (shows the traffic simulation window), open `main.py` and change:

```python
use_gui=False  # change to True
```

---

## Project Structure

```
TrafficMind/
├── agents/
│   ├── dqn.py              # DQN agent
│   └── replay_buffer.py    # Experience replay buffer
├── environment/
│   └── sumo_env.py         # SUMO TraCI environment wrapper
├── sumo/
│   ├── configs/
│   │   ├── basic.sumocfg   # Base simulation config
│   │   ├── peak.sumocfg    # Peak hour config
│   │   └── off_peak.sumocfg
│   ├── net/
│   │   └── grid.net.xml    # Road network
│   └── routes/
│       ├── peak.rou.xml    # Peak hour vehicle routes
│       └── off_peak.rou.xml
├── .env                    # Your local SUMO path (not committed to git)
├── main.py                 # Entry point
├── requirements.txt
└── README.md
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
"<your_sumo_path>\bin\sumo.exe" -c "sumo\configs\basic.sumocfg"
```

---

## Notes

- The `.env` file is intentionally excluded from git (listed in `.gitignore`) since each team member's SUMO path will be different.
- The simulation runs for 3600 steps (1 simulated hour) by default.