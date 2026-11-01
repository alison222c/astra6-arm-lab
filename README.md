# Astra 6 Arm Lab

A Franka Panda arm in a MuJoCo tabletop cell, controlled in natural language by **Astra 6** (an LLM planner: Claude or any OpenAI-compatible model) or by keyboard teleop, in the browser.

The robot works from sensing, not from simulator state:

1. **Perceive** – a fixed RGB-D camera; objects are found by colour, back-projected with depth, and fitted to known shapes (`perception.py`).
2. **Plan** – the LLM gets the labelled camera image plus the estimated positions and returns a JSON action plan (`brain.py`).
3. **Act** – the controller re-looks before grasping, uses a wrist force sensor to detect touchdown when placing, and verifies each step (`robot.py`).

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
.venv/bin/python server.py
```

Then open http://localhost:8000.

The Panda model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache 2.0, see `vendor/menagerie/LICENSE`).
