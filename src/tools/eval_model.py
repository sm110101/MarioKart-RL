import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from stable_baselines3 import DQN

# Allow running this file directly
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.envs.mario_kart_ds_env import MarioKartDSEnv


def make_env() -> MarioKartDSEnv:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = str(repo_root / "ROM" / "mariokart.nds")
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial_t+420.dsv")
    mem_cfg_path = str(repo_root / "src" / "configs" / "memory_addresses.yaml")

    return MarioKartDSEnv(
        rom_path=rom_path,
        savestate_path=savestate_path,
        frame_skip=4,
        memory_config_path=mem_cfg_path,
        settle_frames=0,
        grace_steps_wrong_way=0,
        grace_steps_no_progress=200,
        max_steps=8000,
        debug=True,  # include debug info in info dict
    )


def save_video(frames_rgb: list[np.ndarray], out_path: Path, fps: int = 30) -> None:
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for f in frames_rgb:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main(model_path: Optional[str] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_model = repo_root / "runs" / "dqn_mkds" / "final_model.zip"
    model_path = model_path or str(default_model)

    env = make_env()
    model = DQN.load(model_path, env=None, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

    obs, info = env.reset()
    done, truncated = False, False
    total_reward = 0.0
    frames: list[np.ndarray] = []

    # upscale for video readability
    def upsample(rgb: np.ndarray) -> np.ndarray:
        return cv2.resize(rgb, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += float(reward)
        # Render with overlay like in test_actions
        frame = env.render()
        vis = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
        text = f"r={reward:.3f} spd={info.get('speed')} ww={info.get('wrong_way')} lap={info.get('lap')} col={info.get('collisions_episode')}"
        cv2.putText(vis, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 20), 1, cv2.LINE_AA)
        frames.append(vis)
        # Optional: break early if video gets too long
        if len(frames) >= 30 * 180:  # ~180 seconds at 30 FPS
            break

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = repo_root / "runs" / "videos" / f"eval_{timestamp}.mp4"
    save_video(frames, out_path, fps=30)
    print(f"[eval] episode end: total_reward={total_reward:.3f}, frames={len(frames)}, video={out_path}")
    env.close()
    return 0


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(mp))


