"""Astra 6 Arm Lab server: runs the MuJoCo cell in real time and talks to the browser over a WebSocket.

Run:  .venv/bin/python server.py   then open http://localhost:8000
"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent


def load_dotenv(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
from robot import Cell, Cancelled, ActionError, GRIP_OPEN, GRIP_CLOSED  # noqa: E402
from brain import Brain  # noqa: E402

app = FastAPI()
cell = Cell()
brain = Brain()
clients: set[WebSocket] = set()
geoms = cell.visual_geoms()
scene_msg = json.dumps({"kind": "scene", **cell.scene_json()})
plan_task: asyncio.Task | None = None
CONTROL_DT, SUBSTEPS = 0.01, 5   # 100 Hz control, 500 Hz physics


async def broadcast(msg):
    data = json.dumps(msg) if isinstance(msg, dict) else msg
    for ws in list(clients):
        try:
            if isinstance(data, bytes):
                await ws.send_bytes(data)
            else:
                await ws.send_text(data)
        except Exception:
            clients.discard(ws)


async def physics_loop():
    """Keep simulated time locked to wall-clock time."""
    t0, ticks = time.perf_counter(), 0
    while True:
        due = int((time.perf_counter() - t0) / CONTROL_DT)
        if due - ticks > 10:          # fell behind (machine busy): skip ahead instead of racing
            ticks = due - 10
        while ticks < due:
            cell.control(CONTROL_DT)
            cell.step(SUBSTEPS)
            ticks += 1
        await asyncio.sleep(CONTROL_DT / 2)


async def stream_loop():
    n = 0
    while True:
        if clients:
            await broadcast(cell.frame_bytes(geoms))
            if n % 6 == 0:
                await broadcast({"kind": "status", **cell.status(), "busy": bool(plan_task and not plan_task.done())})
        n += 1
        await asyncio.sleep(1 / 30)


async def perception_loop():
    """Keep the world model fresh and show the browser what the camera sees (~3 Hz)."""
    while True:
        await asyncio.sleep(0.33)
        if cell._look_lock.locked():
            continue                  # the executor is taking its own look right now
        dets = await cell.perceive()
        if clients:
            jpg = cell.vision.annotated_jpeg(quality=70)
            boxes = []
            for oid, det in dets.items():
                # grading only: how far the estimate is from the simulator's true pose
                err = float(np.linalg.norm(det["p"] - cell.obj_pos(oid)) * 1000)
                boxes.append({"id": oid, "box": det["box"], "err_mm": round(err, 1),
                              "p": [round(float(v), 3) for v in det["p"]]})
            await broadcast({"kind": "vision", "jpg": base64.b64encode(jpg).decode(), "dets": boxes,
                             "missing": [o for o in cell.obj_body if o not in dets]})


@app.on_event("startup")
async def start():
    asyncio.create_task(physics_loop())
    asyncio.create_task(stream_loop())
    asyncio.create_task(perception_loop())


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


async def run_command(text, provider, model, base_url):
    await cell.perceive()             # plan from a fresh look
    scene, held = cell.describe()
    image = cell.vision.annotated_jpeg() if cell.sense == "vision" else None
    via = "Claude" if provider != "openai" else "OpenAI-compatible"
    await broadcast({"kind": "thinking", "via": via})
    try:
        plan = await asyncio.to_thread(brain.plan, text, scene, held, provider, model, base_url, image)
    except Exception as e:
        await broadcast({"kind": "error", "message": str(e)})
        return
    await broadcast({"kind": "plan", "reply": plan["reply"], "actions": plan["actions"], "reasoning": plan["reasoning"],
                     "via": f"{via} · {model or 'default'}"})
    if plan["actions"]:
        brain.note_result(await execute(plan["actions"]))


async def execute(actions):
    """Run a plan step by step; returns a one-line outcome for the brain."""
    outcome = f"all {len(actions)} steps succeeded."
    for i, a in enumerate(actions):
        await broadcast({"kind": "step", "i": i, "state": "run"})
        try:
            await cell.run(a)
            await broadcast({"kind": "step", "i": i, "state": "done"})
        except (Cancelled, ActionError) as e:
            msg = "Stopped by operator." if isinstance(e, Cancelled) else str(e)[0].upper() + str(e)[1:] + "."
            await broadcast({"kind": "step", "i": i, "state": "fail", "error": msg})
            outcome = f"step {i + 1} ({json.dumps(a)}) failed: {msg} Later steps were not run."
            break
    cell.cancel = False
    return outcome


def stop_plan():
    if plan_task and not plan_task.done():
        cell.cancel = True
        cell.goal_p = cell.set_p.copy()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global plan_task
    await ws.accept()
    clients.add(ws)
    await ws.send_text(scene_msg)
    await ws.send_text(json.dumps({"kind": "providers", **brain.providers()}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "teleop":
                v = np.array(msg.get("v", [0, 0, 0]), dtype=float)[:3]
                w = float(msg.get("w", 0))
                if np.any(v) or w:
                    stop_plan()
                cell.teleop_v, cell.teleop_w = v * 0.18 * cell.speed, w * 1.2 * cell.speed
            elif t == "grip":
                stop_plan()
                cell.d.ctrl[7] = GRIP_OPEN if msg.get("open") else GRIP_CLOSED
            elif t == "speed":
                cell.speed = float(np.clip(msg.get("value", 1), 0.3, 3))
            elif t == "stop":
                stop_plan()
            elif t == "sense":
                if not (plan_task and not plan_task.done()):
                    cell.sense = "truth" if msg.get("value") == "truth" else "vision"
            elif t in ("reset", "shuffle", "home"):
                stop_plan()
                if plan_task:
                    await asyncio.wait([plan_task], timeout=2)
                if t == "home":
                    plan_task = asyncio.create_task(execute([{"type": "home"}]))
                else:
                    cell.reset() if t == "reset" else cell.shuffle()
                    brain.clear()
            elif t == "command":
                if plan_task and not plan_task.done():
                    await ws.send_text(json.dumps({"kind": "error", "message": "Astra 6 is still running the last plan. Stop it first."}))
                    continue
                plan_task = asyncio.create_task(run_command(str(msg.get("text", ""))[:2000], msg.get("provider", "claude"),
                                                            msg.get("model") or None, msg.get("base_url") or None))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")
