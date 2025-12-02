import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import cv2

# Allow running directly
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.envs.mario_kart_ds_env import MarioKartDSEnv, ACTIONS


def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def run_action(env: MarioKartDSEnv, action_idx: int, steps: int = 240, out_dir: Path | None = None) -> dict:
    obs, info = env.reset()
    # Capture initial top screen
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(env.render(), out_dir / f"action_{action_idx:02d}_start.png")

    total_reward = 0.0
    last_info = {}

    # Live visualization window
    win_name = f"Action {action_idx} - {ACTIONS[action_idx]}"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    for t in range(steps):
        obs, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += reward
        last_info = info

        # Show current top screen with overlayed debug info
        frame = env.render()  # RGB (192,256,3)
        vis = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
        overlay = f"t={t+1} act={ACTIONS[action_idx]} r={reward:.3f} spd={info.get('speed')} prog={info.get('progress_raw')} ww={info.get('wrong_way')}"
        cv2.putText(vis, overlay, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 20), 1, cv2.LINE_AA)
        cv2.imshow(win_name, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # Small wait to keep UI responsive; press 'q' to skip this action early
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print(f"[action {action_idx}] user interrupted at t={t+1}")
            break

        if (t + 1) % 60 == 0:
            print(f"[action {action_idx}] t={t+1} reward={reward} info={info}")
        if terminated or truncated:
            print(f"[action {action_idx}] ended early at t={t+1} (terminated={terminated}, truncated={truncated})")
            break

    if out_dir is not None:
        save_rgb(env.render(), out_dir / f"action_{action_idx:02d}_end.png")

    cv2.destroyWindow(win_name)

    return {
        "steps_run": t + 1,
        "total_reward": total_reward,
        "last_info": last_info,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = str(repo_root / "ROM" / "mariokart.nds")
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial_t+420.dsv")

    # Turn on debug to get info dict populated
    env = MarioKartDSEnv(
        rom_path=rom_path,
        savestate_path=savestate_path,
        frame_skip=4,
        memory_config_path=str(repo_root / "src" / "configs" / "memory_addresses.yaml"),
        settle_frames=0,  # offset savestate
        grace_steps_wrong_way=0,
        grace_steps_no_progress=120,
        max_steps=2000,
        debug=True,
    )

    print("Action mapping:")
    for idx, combo in enumerate(ACTIONS):
        print(f"  {idx}: {combo}")

    out_dir = repo_root / "runs" / "action_checks"
    results = []
    for idx in range(len(ACTIONS)):
        print(f"\n=== Testing action {idx}: {ACTIONS[idx]} ===")
        res = run_action(env, idx, steps=240, out_dir=out_dir)
        results.append((idx, res))

    print("\nSummary:")
    for idx, res in results:
        last = res["last_info"]
        print(
            f"  action {idx}: steps={res['steps_run']}, total_reward={res['total_reward']:.3f}, "
            f"last speed={last.get('speed')}, last progress_raw={last.get('progress_raw')}, "
            f"last wrong_way={last.get('wrong_way')}, last lap={last.get('lap')}"
        )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


