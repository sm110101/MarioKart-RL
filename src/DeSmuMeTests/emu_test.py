from desmume.emulator import DeSmuME
from desmume.controls import Keys
import desmume.controls as controls
import os
import time
import threading


def configure_keyboard_mapping() -> None:
    """
    Configure host keyboard → DS key mappings:
      - Left Arrow  -> D-Pad Left (steer left)
      - Right Arrow -> D-Pad Right (steer right)
      - X           -> A (accelerate)
      - Z           -> B (reverse)
      - Space       -> R (jump; hold with A+steer to drift)

    Notes:
      - Uses desmume.controls module-level mapping helpers.
      - We clear any existing DS-key bindings, then rebind desired host keys.
    """
    # Start from default mapping to ensure a known baseline
    try:
        controls.load_default_config()
    except Exception:
        pass

    # Remove any existing bindings for the DS keys we care about
    for ds_key in (Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_A, Keys.KEY_B, Keys.KEY_R):
        try:
            controls.rm_key(ds_key)
        except Exception:
            pass

    # Rebind: support both lowercase and uppercase for letters on some systems
    bindings = [
        (Keys.KEY_LEFT, ["LEFT", "Left"]),
        (Keys.KEY_RIGHT, ["RIGHT", "Right"]),
        (Keys.KEY_A, ["x", "X"]),
        (Keys.KEY_B, ["z", "Z"]),
        (Keys.KEY_R, ["SPACE", "Space"]),
    ]
    for ds_key, host_keys in bindings:
        for hk in host_keys:
            try:
                controls.add_key(ds_key, hk)
            except Exception:
                pass


def try_load_yoshi_falls_savestate(emu: DeSmuME) -> bool:
    """
    Try to load a savestate that is positioned at the Yoshi Falls
    time trial starting line.
    Expected path: ROM/yoshi_falls_time_trial.dsv

    Returns True if loaded, False otherwise.
    """
    savestate_path = os.path.join("ROM", "yoshi_falls_time_trial.dsv")
    if os.path.exists(savestate_path):
        try:
            emu.savestate.load_file(savestate_path)
            return True
        except Exception:
            return False
    return False


def _idle(emu: DeSmuME, window, frames: int, sleep_sec: float = 0.01) -> None:
    """Run emulator for a number of frames while keeping the window responsive."""
    for _ in range(frames):
        window.process_input()
        time.sleep(sleep_sec)
        emu.cycle(with_joystick=False)
        window.draw()


def _hold_keys(emu: DeSmuME, window, keys: list, frames: int, sleep_sec: float = 0.01) -> None:
    """Hold one or more DS keys for a number of frames, then release."""
    for k in keys:
        try:
            emu.input.keypad_add_key(controls.keymask(k))
        except Exception:
            pass
    try:
        emu.input.keypad_update()
    except Exception:
        pass

    _idle(emu, window, frames, sleep_sec)

    for k in keys:
        try:
            emu.input.keypad_rm_key(controls.keymask(k))
        except Exception:
            pass
    try:
        emu.input.keypad_update()
    except Exception:
        pass

    # Allow a few frames for release to register
    _idle(emu, window, max(2, frames // 6), sleep_sec)


def auto_navigate_time_trial(emu: DeSmuME, window) -> None:
    """
    Automate menu navigation to reach Yoshi Falls Time Trial.
    Sequence provided (interpreting 'click' as A):
      click
      down > click
      click
      click
      down > click
      click
      down > click
      click

    We interleave short idles to allow screens to load.
    """
    # Initial boot wait (logos, etc.)
    _idle(emu, window, frames=180)

    # click
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # down > click
    _hold_keys(emu, window, [Keys.KEY_DOWN], frames=10)
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # click
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # click
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # down > click
    _hold_keys(emu, window, [Keys.KEY_DOWN], frames=10)
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # click
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # down > click
    _hold_keys(emu, window, [Keys.KEY_DOWN], frames=10)
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=40)

    # click
    _hold_keys(emu, window, [Keys.KEY_A], frames=20)
    _idle(emu, window, frames=60)


def start_keyboard_fallback(emu: DeSmuME):
    """
    Optional fallback: hook host keyboard events and directly press/release DS buttons
    using keypad_add_key/keypad_rm_key with proper keymasks.
    Requires the 'keyboard' package; if unavailable, prints instructions.
    Reference API: desmume.emulator.DeSmuME_Input (keypad_add_key/keypad_rm_key).
    """
    try:
        import keyboard  # type: ignore
    except Exception:
        print("Keyboard fallback not active (install with: pip install keyboard).")
        return None

    ds_map = {
        "left": Keys.KEY_LEFT,
        "right": Keys.KEY_RIGHT,
        "x": Keys.KEY_A,
        "z": Keys.KEY_B,
        "space": Keys.KEY_R,
    }

    pressed: set[str] = set()

    def on_event(evt):
        name = evt.name.lower() if getattr(evt, "name", None) else ""
        if name not in ds_map:
            return
        if evt.event_type == "down":
            if name not in pressed:
                pressed.add(name)
                # We don't modify keypad immediately; we set it each frame based on pressed set.
        elif evt.event_type == "up":
            if name in pressed:
                pressed.discard(name)

    # Suppress so SDL doesn't see the same key (prevents wrong default mapping)
    keyboard.hook(on_event, suppress=True)
    print("Keyboard fallback active (SDL suppressed): arrows steer, X accel, Z reverse, Space jump/drift.")

    def build_desired_mask() -> int:
        """Combine currently pressed keys into a single keypad mask."""
        mask = 0
        for name in list(pressed):
            ds_key = ds_map.get(name)
            if ds_key is None:
                continue
            try:
                mask = controls.add_key(mask, controls.keymask(ds_key))
            except Exception:
                pass
        return mask

    return build_desired_mask


def main():
    emu = DeSmuME()
    emu.open('ROM/mariokart.nds')

    # Try to jump directly to Yoshi Falls time trial start using a savestate.
    # If unavailable, auto-navigate menus with a sequence of button presses.
    loaded = try_load_yoshi_falls_savestate(emu)

    # Ensure the emulator is running
    try:
        emu.resume()
    except Exception:
        pass

    window = emu.create_sdl_window()

    # Prefer the OS keyboard fallback to avoid SDL mapping conflicts.
    # If unavailable, fall back to SDL-based mapping.
    build_mask = start_keyboard_fallback(emu)
    if build_mask is None:
        # Configure keyboard mappings for driving controls AFTER window creation
        # so that SDL input is initialized.
        configure_keyboard_mapping()
        # Let one short frame pass to ensure mapping is considered
        try:
            window.process_input()
            emu.cycle(with_joystick=False)
            window.draw()
        except Exception:
            pass

    if not loaded:
        # Attempt scripted navigation; if the UI layout differs, allow user-assisted capture.
        try:
            auto_navigate_time_trial(emu, window)
        except Exception:
            pass

        # Start a background helper to let the user save a savestate precisely at the race start
        # without freezing the emulator window.
        save_requested = {"flag": False}

        def wait_for_user_save():
            print("\nNavigate to the Yoshi Falls Time Trial start line,")
            print("then press ENTER in this console to save a savestate at that exact moment.")
            try:
                input()
                save_requested["flag"] = True
            except Exception:
                pass

        t = threading.Thread(target=wait_for_user_save, daemon=True)
        t.start()
        have_saved = False

    while not window.has_quit():
        window.process_input()
        # If keyboard fallback is active, override keypad to exactly match our mapping each frame
        if build_mask is not None:
            try:
                emu.input.keypad_update(build_mask())
            except Exception:
                pass
        time.sleep(0.01)
        emu.cycle(with_joystick=False)
        window.draw()

        # If user requested a save (pressed Enter in console), persist savestate for future auto-loads
        if not loaded:
            if not have_saved and save_requested.get("flag", False):
                try:
                    os.makedirs("ROM", exist_ok=True)
                    path = os.path.join("ROM", "yoshi_falls_time_trial.dsv")
                    emu.savestate.save_file(path)
                    have_saved = True
                    print(f"Saved savestate to: {path}")
                    print("Restart the script; it will auto-load directly into the race start.")
                except Exception as e:
                    print(f"Failed to save savestate: {e}")


if __name__ == "__main__":
    main()
