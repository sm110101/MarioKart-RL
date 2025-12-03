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
    parser.add_argument("--ckpt-freq", type=int, default=50_000, help="Checkpoint frequency in steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to SB3 model .zip to resume from")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = str(repo_root / "ROM" / "mariokart.nds")
    # Use offset savestate so episode starts measuring immediately
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial_t+480.dsv")
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
        max_steps=1000000,              # long enough to finish a lap comfortably
        debug=False,
    )

    # If watching, enable env debug to expose info used for overlay
    if args.watch:
        env_kwargs["debug"] = True
    # Single-env to start; scale later
    env = DummyVecEnv([make_env(rom_path, savestate_path, mem_cfg_path, frame_skip=4, **env_kwargs)])
    # NOTE: Creating a second DeSmuME instance in-process causes crashes.
    # Avoid an eval_env in the same process; run eval separately after training.

    if args.resume:
        print(f"[train_dqn] Resuming from {args.resume}")
        model = DQN.load(args.resume, env=env, device=device)
    else:
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

    ckpt_callback = CheckpointCallback(save_freq=args.ckpt_freq, save_path=str(log_dir / "ckpts"), name_prefix="dqn_mkds")

    class WatchCallback(BaseCallback):
        def __init__(self, fps: int = 30):
            super().__init__()
            self.win = "Training Viewer"
            self.dt_target = 1.0 / max(1, fps)
            self.last = 0.0
            self.action_counts = None  # lazy-init with action space size

        def _on_training_start(self) -> None:
            cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)

        def _on_step(self) -> bool:
            # show first env only
            try:
                env0 = self.training_env.envs[0]  # type: ignore[attribute-defined-outside-init]
                frame = env0.render()  # RGB (192,256,3)
                game = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
                # Overlay live numbers if available
                info = getattr(env0, "_last_info", {}) or {}
                r = getattr(env0, "_last_reward", None)

                # Maintain rolling action histogram from learner step
                actions = self.locals.get("actions", None)  # vector action per env
                if actions is not None:
                    n_actions = int(self.model.action_space.n)  # type: ignore[attr-defined-outside-init]
                    if self.action_counts is None:
                        self.action_counts = np.zeros(n_actions, dtype=np.int64)
                    # support vec envs
                    for a in np.array(actions).reshape(-1):
                        if 0 <= int(a) < n_actions:
                            self.action_counts[int(a)] += 1
                # Convert to probabilities
                probs = None
                if self.action_counts is not None and self.action_counts.sum() > 0:
                    probs = self.action_counts / self.action_counts.sum()

                # Build side panel
                H, W = game.shape[0], game.shape[1]
                panel_w = 260
                panel = np.ones((H, panel_w, 3), dtype=np.uint8) * 20
                y = 24
                line = 22
                def put(label, value):
                    nonlocal y
                    cv2.putText(panel, f"{label}: {value}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 200), 1, cv2.LINE_AA)
                    y += line
                put("r", f"{(r if r is not None else 0):.3f}")
                put("spd", info.get("speed"))
                put("prog", info.get("progress_raw"))
                put("lap", info.get("lap"))
                put("ww", info.get("wrong_way"))
                put("col", info.get("collisions_episode"))
                rs = info.get("return")
                try:
                    rs_fmt = f"{float(rs):.2f}"
                except Exception:
                    rs_fmt = "0.00"
                put("Rsum", rs_fmt)
                y += 8
                cv2.putText(panel, "Action probs", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 1, cv2.LINE_AA)
                y += line
                # Draw horizontal bars for each action
                if probs is not None:
                    n_actions = len(probs)
                    action_labels = ["Coast", "A", "L", "R", "L+A", "R+A"]
                    for a in range(n_actions):
                        p = float(probs[a])
                        bar_x = 90  # shift bars right so labels are visible
                        usable_w = panel_w - bar_x - 15
                        bar_w = int(usable_w * max(0.0, min(1.0, p)))
                        label = action_labels[a] if a < len(action_labels) else str(a)
                        cv2.putText(panel, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                        cv2.rectangle(panel, (bar_x, y - 12), (bar_x + bar_w, y - 2), (60, 200, 255), thickness=-1)
                        cv2.putText(panel, f"{p:.2f}", (bar_x + bar_w + 5, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                        y += line

                vis = np.hstack([game, panel])
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

    class BestReturnCallback(BaseCallback):
        def __init__(self, save_dir: Path):
            super().__init__()
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.best_return = -1.0

        def _on_step(self) -> bool:
            # Detect episode boundary via dones and query env attribute for cumulative progress
            dones = self.locals.get("dones", None)
            if dones is None:
                return True
            try:
                env0 = self.training_env.envs[0]  # type: ignore[attr-defined-outside-init]
                ep_ret = getattr(env0, "_episode_return", None)
                # Guard: some environments may not expose the attribute yet
                if ep_ret is None:
                    return True
                # When an episode ended in the first (and only) env, evaluate best
                if bool(np.array(dones).reshape(-1)[0]):
                    if isinstance(ep_ret, (int, float)) and float(ep_ret) > self.best_return:
                        self.best_return = float(ep_ret)
                        tag = f"{self.best_return:.2f}".replace(".", "_")
                        save_path = self.save_dir / f"best_return_{tag}_t{self.num_timesteps}.zip"
                        self.model.save(str(save_path))
                        print(f"[train_dqn] New best return {self.best_return:.2f}; saved {save_path}")
            except Exception:
                # Never block training if saving fails
                pass
            return True

    callbacks = [ckpt_callback]
    if args.watch:
        callbacks.append(WatchCallback(fps=args.watch_fps))
    # Always track and save best-return checkpoint (independent of viewer)
    callbacks.append(BestReturnCallback(save_dir=log_dir / "best"))

    total_timesteps = int(args.timesteps)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        # Graceful interrupt: save a snapshot so training can be resumed later
        snap_path = log_dir / "last_model_interrupt"
        model.save(str(snap_path))
        print(f"[train_dqn] Interrupted. Snapshot saved to: {snap_path}.zip")

    model.save(str(log_dir / "final_model"))
    print("[train_dqn] Training complete. Model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


