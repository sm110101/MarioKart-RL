import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import cv2
import yaml

# Allow running directly
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.envs.mario_kart_ds_env import MarioKartDSEnv, ACTIONS


def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def read_mem_value(memory, addr: int, size: int, signed: bool) -> int:
    """
    Robust read using multiple backends:
    - memory.read(start, end, size, signed)
    - memory.unsigned.read(start, end, size)
    - compose from read_u8 bytes
    """
    native_read = getattr(memory, "read", None)
    if callable(native_read):
        return int(native_read(addr, addr, size, signed))
    unsigned = getattr(memory, "unsigned", None)
    u_read = getattr(unsigned, "read", None) if unsigned is not None else None
    if callable(u_read):
        return int(u_read(addr, addr, size))
    read_u8 = getattr(memory, "read_u8", None)
    if callable(read_u8):
        data = bytes(read_u8(addr + i) for i in range(size))
        return int.from_bytes(data, byteorder="little", signed=signed)
    raise RuntimeError("No suitable memory read method found")


def run_action(env: MarioKartDSEnv, action_idx: int, steps: int = 240, out_dir: Path | None = None, mem_cfg: dict | None = None) -> dict:
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

        # Direct memory reads using the same API as probe_memory.py
        mem_vals = {}
        if mem_cfg:
            try:
                m = env.emu.memory  # type: ignore[attr-defined]
                for k in ("progress", "speed", "wrong_way", "lap", "collisions"):
                    entry = mem_cfg.get(k)
                    if not entry:
                        continue
                    addr = int(entry["addr"])
                    size = int(entry["size"])
                    signed = bool(entry.get("signed", False))
                    val = read_mem_value(m, addr, size, signed)
                    scale = float(entry.get("scale", 1.0))
                    mem_vals[k] = val * scale
                    mem_vals[f"{k}_raw"] = val
            except Exception as e:
                mem_vals["err"] = str(e)

        # Show current top screen with overlayed debug info (env info + direct mem)
        frame = env.render()  # RGB (192,256,3)
        vis = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
        overlay = (
            f"t={t+1} act={ACTIONS[action_idx]} r={reward:.3f} "
            f"env[spd={info.get('speed')} prog={info.get('progress_raw')} "
            f"ww={info.get('wrong_way')} col={info.get('collisions_episode')}] "
            f"mem[spd={mem_vals.get('speed')} prog={mem_vals.get('progress')} "
            f"ww={mem_vals.get('wrong_way')} col={mem_vals.get('collisions')}]"
        )
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
    savestate_path = str(repo_root / "ROM" / "yoshi_falls_time_trial_t+480.dsv")
    mem_cfg_path = repo_root / "src" / "configs" / "memory_addresses.yaml"
    mem_cfg = {}
    try:
        with mem_cfg_path.open("r", encoding="utf-8") as f:
            mem_cfg = yaml.safe_load(f) or {}
    except Exception:
        mem_cfg = {}

    # Turn on debug to get info dict populated
    env = MarioKartDSEnv(
        rom_path=rom_path,
        savestate_path=savestate_path,
        frame_skip=4,
        memory_config_path=str(repo_root / "src" / "configs" / "memory_addresses.yaml"),
        settle_frames=0,  # offset savestate
        grace_steps_wrong_way=0,
        grace_steps_no_progress=400,
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
        res = run_action(env, idx, steps=240, out_dir=out_dir, mem_cfg=mem_cfg)
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


