# 2048 — DQN Agent

A compact Deep Q-Network implementation that trains an agent to play 2048.
This repository includes:

- `agent.py` — DQN implementation, replay buffer, training/evaluation helpers.
- `train.py` — training loop, CLI, printing and checkpointing.
- `gui.py` — interactive tkinter viewer with a side panel and AI watch mode.
- `board.py` — 2048 game logic.

Quickstart
----------
1. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies (PyTorch must match your platform; macOS users can use MPS builds):

```bash
pip install numpy torch
```

Tkinter is part of the Python standard library on most platforms; if missing, install your OS package (macOS: included; Linux: `sudo apt install python3-tk`).

Run training (example):

```bash
python train.py --episodes 50000 --window 25 --eval-every 500 --save-every 1000
```

Run the GUI (watch AI or play locally):

```bash
python gui.py
```

Features & Notes
----------------
- Uses a small ConvNet (`agent.TwoZeroFourEightNet`) and a numpy-backed replay buffer for efficiency.
- The GUI provides a speed slider that scales animation durations; set it low to watch the agent at very high speeds (10–20 moves/sec depending on machine).
- A severe retroactive death penalty was added in `agent.py`: it computes a penalty proportional to `k * log2(max_tile)` and seeps a logged penalty into the last `N` pushes (defaults: `k=3.0`, `N=5`). Tune `agent.death_k` and `agent.death_back` if needed.
- The training prints a `TIME` column showing wall-clock time per printed window (useful to benchmark throughput).

Performance tips
----------------
- Use an appropriate PyTorch build for your hardware (CPU, CUDA, or MPS on newer macs).
- Increase `batch_size` to saturate the device, but watch memory usage.
- Increase `learn_every` (e.g., 4 or 8) to collect more transitions between optimizer steps.
- Use the `ReplayBuffer` capacity and `batch_size` tradeoffs to match device throughput.
- If training is still slow, consider moving network evaluation to batched simulation (run multiple environment steps in vectorized form) — this repo uses single-game stepping for clarity.

Tuning
------
- `train.py` exposes CLI flags: `--episodes`, `--window`, `--eval-every`, `--eval-n`, `--save-every`, `--models-dir`.
- For reward shaping, edit `RewardConfig` in `agent.py`.
- To tune death penalty at runtime, change `agent.death_k` and `agent.death_back` in code or add CLI flags in `train.py`.

Files to inspect
----------------
- `agent.py` — learning loop, replay buffer, reward shaping.
- `train.py` — prints, evaluation, checkpointing, and time-per-window measurement.
- `gui.py` — UI scaling (SCALE), animation timing helpers, and speed slider.

License & Attribution
---------------------
This project is provided as-is for experimentation and learning. Add a license file if you share or redistribute.

Questions or next steps
----------------------
If you want, I can:
- Add CLI flags to expose `death_k` and `death_back`.
- Add a `requirements.txt` and a short benchmark script that simulates many games headless to measure throughput.
- Run a short training/eval to verify behavior on your machine.
