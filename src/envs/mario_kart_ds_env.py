from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

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

# Minimal discrete action set; inputs are wired in a later step
ACTIONS: list[list[str]] = [
    [],
    ["A"],
    ["A", "LEFT"],
    ["A", "RIGHT"],
    ["LEFT"],
    ["RIGHT"],
]


# Map abstract button names to py-desmume keypad bitmasks
BUTTON_MAP: dict[str, int] = {
    "A": keymask(Keys.KEY_A),
    "B": keymask(Keys.KEY_B),
    "X": keymask(Keys.KEY_X),
    "Y": keymask(Keys.KEY_Y),
    "L": keymask(Keys.KEY_L),
    "R": keymask(Keys.KEY_R),
    "START": keymask(Keys.KEY_START),
    "SELECT": keymask(Keys.KEY_SELECT),
    "UP": keymask(Keys.KEY_UP),
    "DOWN": keymask(Keys.KEY_DOWN),
    "LEFT": keymask(Keys.KEY_LEFT),
    "RIGHT": keymask(Keys.KEY_RIGHT),
}


class MarioKartDSEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        rom_path: str,
        savestate_path: str,
        frame_skip: int = 4,
        max_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.frame_skip = frame_skip
        self._max_steps = max_steps

        self.emu = DeSmuME()
        self.emu.open(self.rom_path, auto_resume=False)

        self.frame_stacker = FrameStacker(stack_size=4)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, IMG_H, IMG_W), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._episode_steps = 0

    def _load_savestate(self) -> None:
        self.emu.savestate.load_file(self.savestate_path)
        self.emu.resume()

    def _cycle_frames(self, n: int) -> None:
        for _ in range(n):
            self.emu.cycle(with_joystick=True)

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
        for key in BUTTON_MAP.values():
            try:
                self.emu.input.keypad_rm_key(key)
            except Exception:
                # If not pressed, ignore
                pass

    def _apply_buttons_for_frame(self, action: int) -> None:
        # Clear previous inputs to ensure per-frame consistency
        self._clear_action_keys()
        buttons = ACTIONS[action]
        for name in buttons:
            mask = BUTTON_MAP.get(name)
            if mask is not None:
                self.emu.input.keypad_add_key(mask)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_steps = 0
        self._load_savestate()
        # Allow UI elements to settle; adjust if unnecessary
        self._cycle_frames(10)
        first = self._get_obs_single()
        obs = self.frame_stacker.reset(first)
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Apply inputs per frame for consistent "held" semantics
        for _ in range(self.frame_skip):
            self._apply_buttons_for_frame(action)
            self._cycle_frames(1)
        # Ensure keys are released after the action window
        self._clear_action_keys()

        obs_single = self._get_obs_single()
        obs = self.frame_stacker.step(obs_single)

        reward = 0.0  # placeholder; will be replaced with RAM-based shaping
        self._episode_steps += 1
        terminated = False
        truncated = self._episode_steps >= self._max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._capture_top_rgb()

    def close(self) -> None:
        # Destructor will free resources; explicit cleanup can be added if needed
        return


