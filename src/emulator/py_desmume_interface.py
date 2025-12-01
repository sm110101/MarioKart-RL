"""
Mario Kart DS emulator interface for py-desmume.

This wrapper separates emulator specifics from RL env/logic and focuses on:
- Initializing the emulator
- Loading ROM and optional savestate (Yoshi Falls time-trial start)
- Stepping the emulator with discrete DS button combinations
- Reading memory for low-dimensional state features (placeholders)

Imports are guarded so importing this module won't crash on systems
without py-desmume installed. Actual usage will raise a helpful error.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import time

_HAS_DESMUME = True
try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask, add_key, rm_key
except Exception:  # pragma: no cover - guard import for systems without py-desmume
    _HAS_DESMUME = False
    DeSmuME = object  # type: ignore
    Keys = object  # type: ignore
    def keymask(_: int) -> int:  # type: ignore
        return 0
    def add_key(old_mask: int, key: int) -> int:  # type: ignore
        return old_mask
    def rm_key(old_mask: int, key: int) -> int:  # type: ignore
        return old_mask


class MarioKartInterface:
    """
    Lightweight wrapper around py-desmume for Mario Kart DS (Yoshi Falls).

    Parameters
    ----------
    rom_path : str
        Path to Mario Kart DS ROM (e.g., 'ROM/mariokart.nds').
    savestate_path : str | None
        Optional savestate path to start at Yoshi Falls time trial line
        (e.g., 'ROM/yoshi_falls_time_trial.dsv').
    frames_per_step : int
        Number of emulator cycles to run for each action step.
    speed_addr : int | None
        Placeholder memory address for speed (raw).
    lap_addr : int | None
        Placeholder memory address for lap.
    progress_addr : int | None
        Placeholder memory address for progress along lap.
    extra_addrs : dict[str, int] | None
        Optional extra memory addresses to read.
    """

    def __init__(
        self,
        rom_path: str,
        savestate_path: Optional[str] = None,
        frames_per_step: int = 4,
        post_reset_wait_seconds: float = 3.5,
        speed_addr: Optional[int] = None,
        lap_addr: Optional[int] = None,
        progress_addr: Optional[int] = None,
        extra_addrs: Optional[Dict[str, int]] = None,
    ) -> None:
        if not _HAS_DESMUME:
            raise RuntimeError(
                "py-desmume not available. Install and ensure native libs are present to use MarioKartInterface."
            )
        if frames_per_step <= 0:
            raise ValueError("frames_per_step must be positive.")

        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.frames_per_step = frames_per_step
        self.post_reset_wait_seconds = max(0.0, float(post_reset_wait_seconds))
        self.speed_addr = speed_addr
        self.lap_addr = lap_addr
        self.progress_addr = progress_addr
        self.extra_addrs = dict(extra_addrs) if extra_addrs else {}

        self.emu: DeSmuME = DeSmuME()  # type: ignore
        self._opened = False

    def reset(self) -> None:
        """
        Reset emulator session:
        - Open ROM if not opened
        - Load savestate if provided
        - Ensure emulator is running (resume)

        Note: We do not create a window here. Live viewing should be handled
        by the caller using `self.emu.create_sdl_window()` if needed.
        """
        if not self._opened:
            self.emu.open(self.rom_path)
            self._opened = True
        else:
            # Reset current game session (keeping ROM loaded)
            self.emu.reset()

        if self.savestate_path:
            try:
                self.emu.savestate.load_file(self.savestate_path)
            except Exception:
                # If savestate fails to load, proceed from default boot
                pass

        try:
            self.emu.resume()
        except Exception:
            # If resume fails, we will continue but subsequent cycles may no-op
            pass
        # Advance past any countdown/pre-roll so controls are effective immediately (real-time)
        start = time.monotonic()
        while time.monotonic() - start < self.post_reset_wait_seconds:
            try:
                # Ensure keypad is clean during countdown
                self.emu.input.keypad_update(0)
            except Exception:
                pass
            self.emu.cycle(with_joystick=False)

    def step_action(self, action_id: int) -> None:
        """
        Apply a discrete action for `frames_per_step` frames:
        0: no-op
        1: A (accelerate)
        2: A + LEFT
        3: A + RIGHT
        4: A + DOWN (brake/drift)
        5: release / no-op (coast)

        Keys are applied via keypad_update each frame to ensure consistent state.
        """
        mask = 0
        if action_id == 1:
            mask = add_key(mask, keymask(Keys.KEY_A))  # accelerate
        elif action_id == 2:
            mask = add_key(mask, keymask(Keys.KEY_A))
            mask = add_key(mask, keymask(Keys.KEY_LEFT))
        elif action_id == 3:
            mask = add_key(mask, keymask(Keys.KEY_A))
            mask = add_key(mask, keymask(Keys.KEY_RIGHT))
        elif action_id == 4:
            mask = add_key(mask, keymask(Keys.KEY_A))
            mask = add_key(mask, keymask(Keys.KEY_DOWN))
        elif action_id == 5:
            mask = 0  # release / coast
        else:
            mask = 0  # no-op

        for _ in range(self.frames_per_step):
            try:
                self.emu.input.keypad_update(mask)
            except Exception:
                pass
            self.emu.cycle(with_joystick=False)

    def get_state(self) -> Dict[str, Optional[int]]:
        """
        Read available memory fields and return a raw dict.
        Keys:
            - 'lap': int | None
            - 'speed_raw': int | None
            - 'progress_raw': int | None
            - plus any entries from extra_addrs

        TODO:
            - Calibrate addresses and value interpretation (scales, signedness)
            - Consider adding timer or angle if identified
        """
        state: Dict[str, Optional[int]] = {
            "lap": self.get_lap(),
            "speed_raw": self.get_speed_raw(),
            "progress_raw": self.get_progress_raw(),
        }
        # Extras
        for name, addr in self.extra_addrs.items():
            try:
                state[name] = int(self.emu.memory.unsigned[addr])
            except Exception:
                state[name] = None
        return state

    def get_lap(self) -> Optional[int]:
        """Return current lap from memory, or None if not configured."""
        if self.lap_addr is None:
            return None
        try:
            return int(self.emu.memory.unsigned[self.lap_addr])
        except Exception:
            return None

    def get_speed_raw(self) -> Optional[int]:
        """Return raw speed from memory, or None if not configured."""
        if self.speed_addr is None:
            return None
        try:
            return int(self.emu.memory.unsigned[self.speed_addr])
        except Exception:
            return None

    def get_progress_raw(self) -> Optional[int]:
        """Return raw lap progress from memory, or None if not configured."""
        if self.progress_addr is None:
            return None
        try:
            return int(self.emu.memory.unsigned[self.progress_addr])
        except Exception:
            return None


