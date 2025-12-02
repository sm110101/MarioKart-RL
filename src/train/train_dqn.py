import os
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Allow running this file directly
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.envs.mario_kart_ds_env import MarioKartDSEnv
import cv2
import time


def make_env(rom_path: str, savestate_path: str, memory_config_path: str | None = None, frame_skip: int = 4, **env_kwargs):
    def _thunk():
        return MarioKartDSEnv(
            rom_path=rom_path,
            savestate_path=savestate_path,
            frame_skip=frame_skip,
            memory_config_path=memory_config_path,
            **env_kwargs,
        )
    return _thunk


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Show a live window of training episodes")
    parser.add_argument("--watch-fps", type=int, default=30, help="Target FPS for the training viewer")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = str(repo_root / "ROM" / "mariokart.nds")
    # Use offset savestate so episode starts measuring immediately
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial_t+420.dsv")
    mem_cfg_path = str(repo_root / "src" / "configs" / "memory_addresses.yaml")

    log_dir = repo_root / "runs" / "dqn_mkds"
    os.makedirs(log_dir, exist_ok=True)

    # Prefer GPU if available (RTX 4070 Super)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_dqn] Using device: {device}")

    # Production settings: strict wrong-way, immediate measurement (offset savestate), long episodes
    env_kwargs = dict(
        settle_frames=0,              # offset savestate already waited 240 frames
        grace_steps_wrong_way=0,     # terminate immediately on wrong-way
        grace_steps_no_progress=200, # allow brief stalls early
        max_steps=8000,              # long enough to finish a lap comfortably
        debug=False,
    )

    # If watching, enable env debug to expose info used for overlay
    if args.watch:
        env_kwargs["debug"] = True
    # Single-env to start; scale later
    env = DummyVecEnv([make_env(rom_path, savestate_path, mem_cfg_path, frame_skip=4, **env_kwargs)])
    # NOTE: Creating a second DeSmuME instance in-process causes crashes.
    # Avoid an eval_env in the same process; run eval separately after training.

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=10_000,
        train_freq=(4, "step"),
        verbose=1,
        tensorboard_log=str(log_dir),
        device=device,
    )

    # Callbacks: periodic eval and checkpoints
    # Removed EvalCallback to avoid multiple emulator instances in the same process.

    ckpt_callback = CheckpointCallback(save_freq=50_000, save_path=str(log_dir / "ckpts"), name_prefix="dqn_mkds")

    class WatchCallback(BaseCallback):
        def __init__(self, fps: int = 30):
            super().__init__()
            self.win = "Training Viewer"
            self.dt_target = 1.0 / max(1, fps)
            self.last = 0.0

        def _on_training_start(self) -> None:
            cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)

        def _on_step(self) -> bool:
            # show first env only
            try:
                env0 = self.training_env.envs[0]  # type: ignore[attribute-defined-outside-init]
                frame = env0.render()  # RGB (192,256,3)
                vis = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
                # Overlay live numbers if available
                info = getattr(env0, "_last_info", {}) or {}
                r = getattr(env0, "_last_reward", None)
                text = (
                    f"r={r if r is not None else 0:.3f} "
                    f"spd={info.get('speed')} prog={info.get('progress_raw')} "
                    f"ww={info.get('wrong_way')} lap={info.get('lap')} "
                    f"col={info.get('collisions_episode')}"
                )
                cv2.putText(vis, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 20), 1, cv2.LINE_AA)
                now = time.time()
                if now - self.last >= self.dt_target:
                    cv2.imshow(self.win, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    self.last = now
            except Exception:
                pass
            return True

        def _on_training_end(self) -> None:
            try:
                cv2.destroyWindow(self.win)
            except Exception:
                pass

    callbacks = [ckpt_callback]
    if args.watch:
        callbacks.append(WatchCallback(fps=args.watch_fps))

    total_timesteps = int(args.timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save(str(log_dir / "final_model"))
    print("[train_dqn] Training complete. Model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


