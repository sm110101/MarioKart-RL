import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    # py-desmume exposes the emulator under desmume.emulator
    from desmume.emulator import DeSmuME
except Exception as import_error:
    print("Failed to import py-desmume (desmume.emulator). Ensure py-desmume is installed.", file=sys.stderr)
    raise import_error


def resolve_paths() -> tuple[Path, Path]:
    """
    Resolve ROM and savestate paths relative to the repository root.
    This script lives at src/tools/, so the repo root is two levels up.
    """
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = repo_root / "ROM" / "mariokart.nds"
    savestate_path = repo_root / "ROM" / "yoshi_falls_time_trial.dsv"
    return rom_path, savestate_path


def load_emulator(rom_path: Path, savestate_path: Path) -> DeSmuME:
    """
    Initialize emulator, open ROM without auto-resume, and load the savestate.
    """
    emu = DeSmuME()
    # Open ROM but do not start running yet; we control stepping manually
    emu.open(str(rom_path), auto_resume=False)
    # Load the provided savestate to jump directly to time trial start
    emu.savestate.load_file(str(savestate_path))
    # Resume emulation (keypad defaults to released unless keep_keypad=True)
    emu.resume()
    return emu


def step_frames(emu: DeSmuME, frames: int = 30) -> None:
    """
    Advance the emulator by a number of frames.
    """
    for _ in range(frames):
        emu.cycle(with_joystick=True)


def capture_top_screen_fast(emu: DeSmuME) -> np.ndarray:
    """
    Capture the full display buffer via display_buffer_as_rgbx and return the top screen RGB image.
    The DS has two 256x192 screens stacked vertically in the buffer -> (384, 256, 4).
    """
    buf = emu.display_buffer_as_rgbx(reuse_buffer=True)  # memoryview of RGBX
    arr = np.frombuffer(buf, dtype=np.uint8)
    expected_elems = 384 * 256 * 4
    if arr.size != expected_elems:
        raise RuntimeError(f"Unexpected buffer size: {arr.size}, expected {expected_elems}")
    arr = arr.reshape(384, 256, 4)  # H, W, 4
    rgb = arr[..., :3]
    top = rgb[:192, :, :]  # top screen (H=192, W=256, C=3)
    return top


def capture_via_screenshot(emu: DeSmuME) -> np.ndarray:
    """
    Fallback path using screenshot() which returns a PIL image of the stacked screens.
    """
    pil_img = emu.screenshot()
    rgb = np.array(pil_img)  # (384,256,3) in RGB order
    top = rgb[:192, :, :]
    return top


def save_image(arr: np.ndarray, path: Path) -> None:
    """
    Save an RGB uint8 array as PNG.
    """
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def main() -> int:
    rom_path, savestate_path = resolve_paths()

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not savestate_path.exists():
        print(f"Savestate not found: {savestate_path}", file=sys.stderr)
        return 1

    emu = load_emulator(rom_path, savestate_path)

    # Let the state settle a bit (countdown/UI); adjust if not needed
    step_frames(emu, frames=30)

    # Try fast path first
    out_dir = Path(__file__).resolve().parent
    try:
        top_fast = capture_top_screen_fast(emu)
        print(f"Fast path top screen shape: {top_fast.shape} (H,W,C)")
        save_image(top_fast, out_dir / "savestate_start_fast.png")
    except Exception as fast_err:
        print(f"Fast path failed ({fast_err}); falling back to screenshot()", file=sys.stderr)
        top_scr = capture_via_screenshot(emu)
        print(f"screenshot() top screen shape: {top_scr.shape} (H,W,C)")
        save_image(top_scr, out_dir / "savestate_start_screenshot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

