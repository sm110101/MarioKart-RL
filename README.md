## MarioKart-RL (Mario Kart DS with Reinforcement Learning)

Train a DQN agent to drive in Mario Kart DS using pixel observations and RAM‑based rewards, built on top of the `py-desmume` Nintendo DS emulator and a custom Gymnasium environment.

### Demo

The agent successfully completes a full lap on Yoshi Falls. Watch a sample run below:

![Full lap demo](img/Full_lap.gif)

### What’s inside

- **Gym environment**: `MarioKartDSEnv` wraps `py-desmume` (ROM loading, savestates, input, memory reads).
- **Observations**: Atari-style preprocessing (crop top screen, grayscale, resize to 84×84) with 4‑frame stacking.
- **Actions**: Discrete mapping using DS keypad masks mirrored from working PC keys.
- **Rewards/Termination**: RAM-driven shaping (speed, lap progress), strict penalties for wrong‑way and collisions, speed sanitization.
- **Training**: SB3 DQN (`CnnPolicy`) with CUDA support, checkpoints, and a live viewer overlay (metrics + action probs).
- **Tools**: Scripts to verify savestates, probe memory, test actions, manual control, train, and evaluate.

### Quick start

1) Python

- Windows 10/11. Recommended: Python 3.12 (GPU training uses CUDA PyTorch builds).

2) Install

```bash
pip install -r requirements.txt
# (Optional, GPU) Install a CUDA build of PyTorch per https://pytorch.org/get-started/locally/
```

3) Verify ROM + savestate

```bash
python -m src.tools.verify_savestate
```

You should see a saved screenshot from the top screen and no errors.

### Train

```bash
python -m src.train.train_dqn \
  --timesteps 1000000 \
  --logdir runs/dqn_mkds \
  --ckpt-freq 50000 \
  --watch --watch-fps 30
```

- Uses GPU automatically when available.
- Best‑return checkpoint is kept in `runs/dqn_mkds/best/best_return.zip`.
- To resume from a checkpoint:

```bash
python -m src.train.train_dqn --resume runs/dqn_mkds/checkpoints/ckpt_500000_steps.zip
```

### Evaluate

```bash
python -m src.tools.eval_model --model runs/dqn_mkds/best/best_return.zip
```

Generates a short video with the same on‑screen metrics as the training viewer.

### Configuration notes

- ROMs/savestates: place in `ROM/` (e.g., `ROM/mariokart.nds`, `ROM/yoshi_falls_time_trial.dsv`).
- Memory addresses used for reward/termination live in `src/configs/memory_addresses.yaml`.
- Only one emulator instance should run in the process (training script enforces this).

### Acknowledgments

- `py-desmume` and DeSmuME developers.
- Stable‑Baselines3 team for DQN.


