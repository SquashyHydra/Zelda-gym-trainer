# Zelda: Link's Awakening — Trainer & Gym Environment

This repository contains tools and a Gym-style environment for training and running agents on the Game Boy ROM "The Legend of Zelda: Link's Awakening" (cartridge file in `Rom/`). It includes scripts for running pretrained agents, collecting sessions, and running experiments with PyBoy-based runners.

## Overview
- `ZeldaGym/` — Gym environment and helper modules (actions, memory addresses, rewards, etc.).
- `pyboyrunner.py` — Runner that interacts with the emulator.
- `run_pretrained_interactive.py` — Launches a pretrained agent in interactive mode.
- `run_parallel_fast.py` — Utilities to run experiments in parallel.
- `Requirements` — Python dependencies are listed in `requirements.txt`.

## Requirements
- Python 3.8+ recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup
1. Place the ROM file in `Rom/` (the file should be named `Zelda.gb`).
2. (Optional) Place or create session/state files in `Sessions/` and `States/` as needed.

## Running
- Run the interactive pretrained agent:

```bash
python run_pretrained_interactive.py
```

- Run the generic runner:

```bash
python pyboyrunner.py
```

- Run parallel experiments (example):

```bash
python run_parallel_fast.py
```

See the top of each script for additional CLI options and usage. You can also inspect `argparse_zelda.py` and `help_message.py` for script argument patterns.

## Project structure

- `ZeldaGym/` — environment implementation
  - `env.py` — Gym env wrapper
  - `modules/` — domain-specific helpers (`actions.py`, `mem_addresses.py`, `rewards.py`, etc.)
- `Rom/` — contains `Zelda.gb` and RAM/state files
- `Sessions/` — recorded session data and pretrained sessions
- `States/` — sample state files used by scripts

## Contributing
- Open an issue for feature requests or bug reports.
- PRs should include focused changes and, where applicable, small tests or a short README update.
- Repository no longer maintained by owner