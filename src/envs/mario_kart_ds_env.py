from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from pathlib import Path

from src.vision.preprocess import preprocess_frame, FrameStacker, IMG_H, IMG_W

try:
    from desmume.emulator import DeSmuME
except Exception as import_error:
    raise import_error

try:
    # Keys enum and keymask for keypad/button mapping
    from desmume.controls import Keys, keymask
except Exception as import_error:
    raise import_error

# Minimal discrete action set; expressed using PC key names
ACTIONS: list[list[str]] = [
    [],                # 0: no input
    ["z"],             # 1: accelerate (Z -> A)
    ["left"],          # 2: steer left
    ["right"],         # 3: steer right
    ["z", "left"],     # 4: accelerate + left (no jump/drift)
    ["z", "right"],    # 5: accelerate + right (no jump/drift)
]


# Map PC keys to DS keypad masks (aligned with manual_control.py)
PC_TO_DS: dict[str, int] = {
    "z": keymask(Keys.KEY_A),
    "x": keymask(Keys.KEY_B),
    "s": keymask(Keys.KEY_X),
    "a": keymask(Keys.KEY_Y),
    "q": keymask(Keys.KEY_L),
    "w": keymask(Keys.KEY_R),
    "up": keymask(Keys.KEY_UP),
    "down": keymask(Keys.KEY_DOWN),
    "left": keymask(Keys.KEY_LEFT),
    "right": keymask(Keys.KEY_RIGHT),
    "enter": keymask(Keys.KEY_START),
    "backspace": keymask(Keys.KEY_SELECT),
}


class MarioKartDSEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        rom_path: str,
        savestate_path: str,
        frame_skip: int = 4,
        max_steps: int = 2000,
        memory_config_path: str | None = None,
        settle_frames: int = 10,
        grace_steps_wrong_way: int = 60,
        grace_steps_no_progress: int = 60,
        debug: bool = False,
        debug_log_path: str | None = None,
    ) -> None:
        super().__init__()
        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.frame_skip = frame_skip
        self._max_steps = max_steps
        self._warned_no_memory = False
        self._settle_frames = max(0, int(settle_frames))
        self._grace_steps_wrong_way = max(0, int(grace_steps_wrong_way))
        self._grace_steps_no_progress = max(0, int(grace_steps_no_progress))
        self._debug = bool(debug)
        self._debug_log_path = debug_log_path

        self.emu = DeSmuME()
        self.emu.open(self.rom_path, auto_resume=False)

        self.frame_stacker = FrameStacker(stack_size=4)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, IMG_H, IMG_W), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._episode_steps = 0

        # Memory config for reward/termination
        if memory_config_path is None:
            # Default to repo's config path
            repo_root = Path(__file__).resolve().parents[2]
            memory_config_path = str(repo_root / "src" / "configs" / "memory_addresses.yaml")
        self._mem_cfg = self._load_memory_config(memory_config_path)
        # Track both raw and scaled progress for robust delta under wrap-around
        self._prev_progress_raw: int | None = None
        self._prev_progress: float | None = None
        self._last_lap: int | None = None
        self._no_progress_counter: int = 0
        self._since_reset_steps: int = 0
        self._collisions_prev_raw: int | None = None
        self._collisions_count: int = 0
        # If frame_skip=4, ~15 env steps ≈ 1 second at 60 FPS
        self._no_progress_limit_steps: int = max(1, int((60 // max(1, frame_skip)) * 5))  # ~5 seconds
        # Pre-compute modulus for progress unwrapping based on configured size (bytes)
        progress_entry = self._mem_cfg.get("progress", {})
        self._progress_size_bytes: int = int(progress_entry.get("size", 2)) or 2
        self._progress_modulus: int = 1 << (8 * self._progress_size_bytes)
        # Collisions modulus for delta
        coll_entry = self._mem_cfg.get("collisions", {})
        self._collisions_size_bytes: int = int(coll_entry.get("size", 1)) or 1
        self._collisions_modulus: int = 1 << (8 * self._collisions_size_bytes)

        # Collision handling parameters
        self._collision_penalty: float = float(0.5)
        self._collision_terminate_threshold: int = int(3)

    def _load_savestate(self) -> None:
        self.emu.savestate.load_file(self.savestate_path)
        self.emu.resume()

    def _cycle_frames(self, n: int) -> None:
        for _ in range(n):
            # We didn't initialize joystick processing; avoid overriding keypad state
            self.emu.cycle(with_joystick=False)

    def _capture_top_rgb(self) -> np.ndarray:
        # Fast path
        buf = self.emu.display_buffer_as_rgbx(reuse_buffer=True)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(384, 256, 4)
        rgb = arr[..., :3]
        top = rgb[:192, :, :]
        return top

    def _get_obs_single(self) -> np.ndarray:
        top_rgb = self._capture_top_rgb()
        return preprocess_frame(top_rgb)

    def _clear_action_keys(self) -> None:
        # Release all keys we may have pressed for control actions
        for key in PC_TO_DS.values():
            try:
                self.emu.input.keypad_rm_key(key)
            except Exception:
                # If not pressed, ignore
                pass

    def _press_action_keys(self, action: int) -> None:
        # Set keys for this action (held across multiple frames)
        buttons = ACTIONS[action]
        for name in buttons:
            mask = PC_TO_DS.get(name.lower())
            if mask is not None:
                self.emu.input.keypad_add_key(mask)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_steps = 0
        self._load_savestate()
        # Allow UI elements to settle; adjust if unnecessary
        if self._settle_frames > 0:
            self._cycle_frames(self._settle_frames)
        first = self._get_obs_single()
        obs = self.frame_stacker.reset(first)
        # Reset reward/termination trackers
        self._prev_progress_raw = self._read_entry_raw("progress")
        self._prev_progress = self._read_progress()
        self._last_lap = self._read_lap()
        self._no_progress_counter = 0
        self._since_reset_steps = 0
        self._collisions_prev_raw = self._read_entry_raw("collisions")
        self._collisions_count = 0
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Hold inputs across the entire frame_skip window
        self._press_action_keys(action)
        self._cycle_frames(self.frame_skip)
        # Release inputs after the window
        self._clear_action_keys()

        obs_single = self._get_obs_single()
        obs = self.frame_stacker.step(obs_single)

        reward = self._compute_reward()
        self._episode_steps += 1
        self._since_reset_steps += 1
        terminated = self._check_terminated()
        truncated = self._episode_steps >= self._max_steps
        info: dict[str, Any] = {}
        if self._debug:
            info["progress_raw"] = self._read_entry_raw("progress")
            info["speed"] = self._read_speed()
            info["wrong_way"] = self._read_wrong_way()
            info["lap"] = self._read_lap()
            info["collisions_raw"] = self._read_entry_raw("collisions")
            info["collisions_episode"] = self._collisions_count
        # make last info available to viewers
        self._last_info = info  # type: ignore[assignment]
        self._last_reward = reward  # type: ignore[assignment]
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._capture_top_rgb()

    def close(self) -> None:
        # Destructor will free resources; explicit cleanup can be added if needed
        return

    # -------- Memory-backed Reward / Termination ----------
    def _load_memory_config(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        return data

    def _read_entry(self, name: str) -> float | int | None:
        cfg = self._mem_cfg.get(name)
        if not cfg:
            if not self._warned_no_memory:
                print(f"[MarioKartDSEnv] Missing memory config for '{name}'. Reward will be 0.", flush=True)
                self._warned_no_memory = True
            return None
        try:
            addr = int(cfg["addr"])
            size = int(cfg["size"])
            little_endian = bool(cfg.get("little_endian", True))
            signed = bool(cfg.get("signed", False))
            mem = self.emu.memory
            value: int | None = None
            # 1) Primary documented API
            native_read = getattr(mem, "read", None)
            if callable(native_read):
                value = int(native_read(addr, addr, size, signed))
            # 2) Unsigned accessor (some builds expose MemoryAccessor)
            if value is None:
                unsigned = getattr(mem, "unsigned", None)
                u_read = getattr(unsigned, "read", None) if unsigned is not None else None
                if callable(u_read):
                    value = int(u_read(addr, addr, size))
            # 3) Fallback: compose from u8
            if value is None:
                read_u8 = getattr(mem, "read_u8", None)
                if callable(read_u8):
                    data = bytes(read_u8(addr + i) for i in range(size))
                    byteorder = "little" if little_endian else "big"
                    value = int.from_bytes(data, byteorder=byteorder, signed=signed)
            if value is None:
                raise RuntimeError("No suitable memory read method found")
            scale = float(cfg.get("scale", 1.0))
            return value * scale
        except Exception:
            if not self._warned_no_memory:
                print(f"[MarioKartDSEnv] Failed reading '{name}' from memory. Reward will be 0.", flush=True)
                self._warned_no_memory = True
            return None

    def _read_entry_raw(self, name: str) -> int | None:
        """Read an entry as an unsigned integer ignoring any scale; useful for wrap-around logic."""
        cfg = self._mem_cfg.get(name)
        if not cfg:
            return None
        try:
            addr = int(cfg["addr"])
            size = int(cfg["size"])
            mem = self.emu.memory
            # 1) Primary API
            native_read = getattr(mem, "read", None)
            if callable(native_read):
                return int(native_read(addr, addr, size, False))
            # 2) Unsigned accessor
            unsigned = getattr(mem, "unsigned", None)
            u_read = getattr(unsigned, "read", None) if unsigned is not None else None
            if callable(u_read):
                return int(u_read(addr, addr, size))
            # 3) Fallback from bytes
            read_u8 = getattr(mem, "read_u8", None)
            if callable(read_u8):
                data = bytes(read_u8(addr + i) for i in range(size))
                return int.from_bytes(
                    data,
                    byteorder="little" if bool(cfg.get("little_endian", True)) else "big",
                    signed=False,
                )
            raise RuntimeError("No suitable memory read method found")
        except Exception:
            return None

    def _read_progress(self) -> float | None:
        return self._read_entry("progress")

    def _read_speed(self) -> float | None:
        return self._read_entry("speed")

    def _read_wrong_way(self) -> int | None:
        # Prefer explicit wrong_way entry; fall back to legacy off_road key
        val = self._read_entry("wrong_way")
        if val is None:
            val = self._read_entry("off_road")
        if val is None:
            return None
        return int(val != 0)

    def _read_collisions_raw(self) -> int | None:
        return self._read_entry_raw("collisions")

    def _read_lap(self) -> int | None:
        val = self._read_entry("lap")
        return int(val) if val is not None else None

    def _compute_reward(self) -> float:
        # Speed-based reward with collision penalties.
        speed = self._read_speed()
        wrong_way = self._read_wrong_way()

        # Collision delta (handle wrap for 1-byte counters)
        coll_raw = self._read_collisions_raw()
        if coll_raw is not None:
            if self._collisions_prev_raw is None:
                self._collisions_prev_raw = coll_raw
            delta = (int(coll_raw) - int(self._collisions_prev_raw)) % self._collisions_modulus
            if delta > 0:
                self._collisions_count += delta
            self._collisions_prev_raw = coll_raw

        # Handle missing readings gracefully
        spd = float(speed or 0.0)
        ww = 1 if (wrong_way is not None and wrong_way != 0) else 0

        # Consider "stuck" if speed stays below a tiny threshold
        if spd <= 1e-3:
            self._no_progress_counter += 1
        else:
            self._no_progress_counter = 0

        # Reward: proportional to speed, penalty for wrong-way and collisions
        r = 0.05 * spd  # tune coefficient as needed to avoid clipping
        if ww:
            r -= 0.1
        if self._collisions_count > 0 and coll_raw is not None:
            # Penalize only on delta this step
            # (recompute delta for reward to avoid double-penalizing)
            delta = (int(coll_raw) - int(self._collisions_prev_raw or coll_raw)) % self._collisions_modulus
            if delta > 0:
                r -= self._collision_penalty * float(delta)
        return float(np.clip(r, -1.0, 1.0))

    def _check_terminated(self) -> bool:
        # Terminate immediately on wrong-way flag (after grace period)
        ww = self._read_wrong_way()
        if ww is not None and ww != 0 and self._since_reset_steps > self._grace_steps_wrong_way:
            return True

        lap = self._read_lap()
        if lap is not None:
            if self._last_lap is None:
                self._last_lap = lap
            elif lap != self._last_lap:
                # Lap changed -> terminate episode
                self._last_lap = lap
                return True
        # Stuck detection: no progress for ~N seconds
        if (self._since_reset_steps > self._grace_steps_no_progress and
                self._no_progress_counter >= self._no_progress_limit_steps):
            return True
        # Collision termination
        if self._collisions_count >= self._collision_terminate_threshold:
            return True
        return False


