"""
server.py FastAPI backend for the TrafficMind dashboard.

Start with:
    uvicorn server:app --port 8080

"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TrafficMind Versus")


app.mount("/static", StaticFiles(directory="static"), name="static")
DASHBOARD_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = DASHBOARD_DIR.parent.resolve()
RUNNER_PATH = DASHBOARD_DIR / "runner.py"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
app.mount("/plots", StaticFiles(directory=str(PROJECT_ROOT / "output_plots")), name="plots")

LIVE_A = str(DASHBOARD_DIR / "live_a.json")
LIVE_B = str(DASHBOARD_DIR / "live_b.json")
# ── Serve dashboard HTML ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the home page."""
    html_path = Path(__file__).parent / "templates/home.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/comparison", response_class=HTMLResponse)
async def read_comparison_page():
    """Serve the comparison page."""
    html_path = Path(__file__).parent / "templates/comparison.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/results", response_class=HTMLResponse)
async def read_comparison_page():
    """Serve the results page."""
    html_path = Path(__file__).parent / "templates/results.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/training_graph", response_class=HTMLResponse)
async def read_training_graph():
    """Serve the training graph page."""
    html_path = Path(__file__).parent / "templates/training_graph.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Launch endpoint ────────────────────────────────────────────────────────────

@app.post("/run")
async def run_versus(payload: dict):
    """Launch two SUMO simulations for comparison."""
    """
    Expected payload:
    {
        "agent_a":      "trained_dqn",
        "models_dir_a": "outputs/local_peak",   # ignored if not dqn
        "agent_b":      "fixed_time",
        "models_dir_b": null,
        "reward":       "local",
        "scenario":     "peak",
        "seed":         42
    }
    """
    # Clear old live files
    for f in [LIVE_A, LIVE_B]:
        if os.path.exists(f):
            os.remove(f)

    python = sys.executable
    base_cmd = [
        python, "runner.py",
        "--reward",   payload["reward"],
        "--scenario", payload["scenario"],
        "--seed",     str(payload.get("seed", 42)),
        "--gui",
    ]

    # Agent A
    cmd_a = base_cmd + [
        "--agent",     payload["agent_a"],
        "--live-file", LIVE_A,
        "--slot",      "a",
    ]
    if payload["agent_a"] == "trained_dqn" and payload.get("models_dir_a"):
        cmd_a += ["--models-dir", payload["models_dir_a"]]

    # Agent B
    cmd_b = base_cmd + [
        "--agent",     payload["agent_b"],
        "--live-file", LIVE_B,
        "--slot",      "b",
    ]
    if payload["agent_b"] == "trained_dqn" and payload.get("models_dir_b"):
        cmd_b += ["--models-dir", payload["models_dir_b"]]

    subprocess.Popen(cmd_a, cwd=str(DASHBOARD_DIR))
    # Small delay so the two SUMO GUIs don't fight over the same port
    await asyncio.sleep(1.5)
    subprocess.Popen(cmd_b, cwd=str(DASHBOARD_DIR))

    return {"status": "launched"}


# ── WebSocket: stream live stats to browser ────────────────────────────────────

def read_live(path: str) -> Optional[dict]:
    """Read live stats from a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming live stats."""
    await websocket.accept()
    try:
        while True:
            data_a = read_live(LIVE_A)
            data_b = read_live(LIVE_B)

            await websocket.send_json({
                "a": data_a,
                "b": data_b,
            })

            # Stop pushing once both are done
            a_done = data_a is not None and data_a.get("status") == "done"
            b_done = data_b is not None and data_b.get("status") == "done"
            if a_done and b_done:
                await websocket.send_json({"finished": True, "a": data_a, "b": data_b})
                break

            await asyncio.sleep(0.5)   # update every 500ms

    except WebSocketDisconnect:
        pass


# ── Discover model folders (for dashboard dropdowns) ──────────────────────────

@app.get("/models")
def list_models():
    """List available model directories."""
    outputs = Path(__file__).parent.parent.resolve() / "outputs"
    if not outputs.exists():
        return {"folders": []}
    folders = [
        d.name for d in sorted(outputs.iterdir())
        if d.is_dir()
        and d.name != "outputs"
        and any(d.glob("agent_*_final.pth"))
    ]
    return {"folders": folders}



# uvicorn server:app --port 8080