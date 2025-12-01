import sys
from pathlib import Path

import numpy as np

# Allow running this file directly
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.envs.mario_kart_ds_env import MarioKartDSEnv


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = str(repo_root / "ROM" / "mariokart.nds")
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial.dsv")

    env = MarioKartDSEnv(rom_path=rom_path, savestate_path=savestate_path, frame_skip=4)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}, dtype={obs.dtype}")

    steps = 100
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (t + 1) % 20 == 0:
            print(f"t={t+1}: obs shape={obs.shape}, reward={reward}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            print(f"Episode ended at t={t+1} (terminated={terminated}, truncated={truncated})")
            break

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

