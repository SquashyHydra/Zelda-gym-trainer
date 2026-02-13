# Zelda: Link's Awakening — Trainer & Gym Environment

This project contains a Gymnasium-compatible environment and training scripts for learning to play the Game Boy game **The Legend of Zelda: Link's Awakening** with Stable-Baselines3 PPO.

## What is included
- `ZeldaGym/` — custom environment, rewards, memory mapping, media helpers
- `run_parallel_fast.py` — main PPO training entrypoint
- `run_pretrained_interactive.py` — run a trained checkpoint interactively
- `pyboyrunner.py` — direct emulator inspection utility
- `argparse_zelda.py` — CLI options used across scripts

## Requirements
- Python 3.10+
- ROM file at `Rom/Zelda.gb`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start
1. Ensure `Rom/Zelda.gb` exists.
2. Start a short training run:

```bash
python run_parallel_fast.py --headless True --process_num 1 --train_timesteps 4096
```

3. Check artifacts in your session folder under `Sessions/` (`PPO_*`, videos, images, stats).

## Training examples

### 1) Train by timesteps
Use this when you want an explicit total sample budget.

```bash
python run_parallel_fast.py --headless True --process_num 2 --train_timesteps 200000 --save_video False --print_rewards False
```

### 2) Train by iterations (rollouts)
Each iteration is `iteration_steps * process_num` samples.

```bash
python run_parallel_fast.py --headless True --process_num 4 --iteration_steps 4096 --train_iterations 20 --save_video False --print_rewards False
```

### 3) Resume from checkpoint
Point `--checkpoint` to the session folder that contains `zelda_*_steps.zip` files.

```bash
python run_parallel_fast.py --headless True --process_num 4 --checkpoint Sessions/session_xxxxxxxx --train_iterations 10
```

### 4) Run with visible emulator window (debug)

```bash
python run_parallel_fast.py --headless False --process_num 1 --train_timesteps 4096 --print_rewards True
```

## Interactive inference example
Run a trained checkpoint in interactive mode:

```bash
python run_pretrained_interactive.py --headless False --checkpoint Sessions/session_xxxxxxxx --save_video True
```

## Most useful CLI flags
- `--process_num` (`-pn`): number of parallel envs
- `--iteration_steps` (`-is`): PPO rollout length (`n_steps`) per iteration
- `--train_iterations` (`-ti`): number of rollout iterations
- `--train_timesteps` (`-tt`): explicit total timesteps (used when `train_iterations=0`)
- `--session_path`: output folder (auto-generated if not set)
- `--checkpoint`: session folder to resume from
- `--curriculum_enabled`: enable phase-based reset states
- `--action_penalty_scale`: action-mask penalty scaling

### Timesteps precedence
`run_parallel_fast.py` computes `total_timesteps` in this order:
1. `train_iterations > 0` → `train_iterations * iteration_steps * process_num`
2. else `train_timesteps > 0` → `train_timesteps`
3. else one rollout (`iteration_steps * process_num`)

## Performance tips
- For best throughput: `--headless True --save_video False --print_rewards False`
- Start with `--process_num` near your available CPU cores, then tune
- Increase `--iteration_steps` for longer PPO rollouts before each policy update

## Project layout
- `ZeldaGym/env.py` — environment loop and observation/reward flow
- `ZeldaGym/modules/rewards.py` — reward shaping logic
- `ZeldaGym/modules/mem_addresses.py` — RAM map constants
- `States/` — state files used as reset starts
- `Sessions/` — outputs: checkpoints, TensorBoard logs, stats, media

## Notes
- This repository is currently community-maintained.
- If training behavior changes unexpectedly, compare your run flags and checkpoint path first.