import sys
import time
from pathlib import Path
from typing import Dict, Set

import cv2
import numpy as np
from pynput import keyboard

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except Exception as import_error:
    print("Failed to import py-desmume. Ensure py-desmume is installed.", file=sys.stderr)
    raise import_error


# Map PC keys to DS keypad masks
PC_TO_DS: Dict[str, int] = {
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


def resolve_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = repo_root / "ROM" / "mariokart.nds"
    # Use offset savestate so gameplay starts immediately
    savestate_path = repo_root / "ROM" / "yoshi_falls_time_trial_t+420.dsv"
    return rom_path, savestate_path


def buffer_to_top_rgb(emu: DeSmuME) -> np.ndarray:
    buf = emu.display_buffer_as_rgbx(reuse_buffer=True)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(384, 256, 4)
    return arr[:192, :, :3]


def start_keyboard_listener(pressed: Set[str]):
    def on_press(key):
        try:
            name = key.char.lower()
        except AttributeError:
            name = str(key).replace("Key.", "")
        if name in PC_TO_DS:
            pressed.add(name)
        if name == "esc":
            # ESC closes the app
            pressed.add("esc")
        return True

    def on_release(key):
        try:
            name = key.char.lower()
        except AttributeError:
            name = str(key).replace("Key.", "")
        if name in pressed:
            pressed.discard(name)
        return True

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def apply_pressed(emu: DeSmuME, pressed: Set[str]) -> None:
    # Clear all known keys, then add currently pressed
    for mask in PC_TO_DS.values():
        try:
            emu.input.keypad_rm_key(mask)
        except Exception:
            pass
    for name in pressed:
        if name == "esc":
            continue
        mask = PC_TO_DS.get(name)
        if mask is not None:
            emu.input.keypad_add_key(mask)


def main() -> int:
    rom_path, savestate_path = resolve_paths()

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not savestate_path.exists():
        print(f"Savestate not found: {savestate_path}", file=sys.stderr)
        return 1

    print("Controls:")
    print("  Drive: Z(A), A(Y), arrows for D-Pad; Q(L), W(R)")
    print("  Start: Enter, Select: Backspace")
    print("  ESC to quit, R to reload savestate, P to pause/resume")

    emu = DeSmuME()
    emu.open(str(rom_path), auto_resume=False)
    emu.savestate.load_file(str(savestate_path))
    emu.resume()

    pressed: Set[str] = set()
    listener = start_keyboard_listener(pressed)

    paused = False
    window = "MKDS Manual Control (Top Screen)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    try:
        last_time = time.time()
        while True:
            # Hotkeys that don't map to DS buttons
            if "p" in pressed:
                paused = not paused
                pressed.discard("p")
                if paused:
                    emu.pause()
                else:
                    emu.resume()

            if "r" in pressed:
                pressed.discard("r")
                emu.pause()
                emu.savestate.load_file(str(savestate_path))
                emu.resume()

            if "esc" in pressed:
                break

            if not paused:
                apply_pressed(emu, pressed)
                emu.cycle(with_joystick=False)

            frame = buffer_to_top_rgb(emu)
            # Optional resize for viewing
            vis = cv2.resize(frame, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            # Small wait to keep UI responsive (~60 FPS)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                paused = not paused

            # simple fps limiter to avoid pegging CPU
            now = time.time()
            dt = now - last_time
            if dt < (1.0 / 120.0):
                time.sleep((1.0 / 120.0) - dt)
            last_time = now
    finally:
        listener.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


