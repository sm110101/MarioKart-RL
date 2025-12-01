import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from desmume.emulator import DeSmuME
except Exception as import_error:
    print("Failed to import py-desmume (desmume.emulator). Ensure py-desmume is installed.", file=sys.stderr)
    raise import_error

# Allow running this file directly (python src/tools/verify_preproc.py)
# by adding the repository root to sys.path so `import src.*` works.
if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.vision.preprocess import preprocess_frame, FrameStacker


def resolve_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = repo_root / "ROM" / "mariokart.nds"
    savestate_path = repo_root / "ROM" / "yoshi_falls_time_trial.dsv"
    out_dir = Path(__file__).resolve().parent
    return rom_path, savestate_path, out_dir


def load_emulator(rom_path: Path, savestate_path: Path) -> DeSmuME:
    emu = DeSmuME()
    emu.open(str(rom_path), auto_resume=False)
    emu.savestate.load_file(str(savestate_path))
    emu.resume()
    return emu


def step_frames(emu: DeSmuME, frames: int = 10) -> None:
    for _ in range(frames):
        emu.cycle(with_joystick=True)


def capture_top_screen(emu: DeSmuME) -> np.ndarray:
    # Prefer fast buffer path
    try:
        buf = emu.display_buffer_as_rgbx(reuse_buffer=True)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(384, 256, 4)
        rgb = arr[..., :3]
        top = rgb[:192, :, :]
        return top
    except Exception:
        # Fallback to screenshot()
        pil_img = emu.screenshot()
        rgb = np.array(pil_img)  # (384,256,3)
        return rgb[:192, :, :]


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


def main() -> int:
    rom_path, savestate_path, out_dir = resolve_paths()

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not savestate_path.exists():
        print(f"Savestate not found: {savestate_path}", file=sys.stderr)
        return 1

    emu = load_emulator(rom_path, savestate_path)
    step_frames(emu, frames=10)

    top_rgb = capture_top_screen(emu)
    print(f"Top RGB shape: {top_rgb.shape} (expected ~ (192,256,3))")
    save_image(top_rgb, out_dir / "preproc_top_raw.png")

    proc = preprocess_frame(top_rgb)
    print(f"Preprocessed shape: {proc.shape} (expected (84,84)) dtype={proc.dtype}")
    # Save grayscale as PNG (convert to RGB for visualization)
    Image.fromarray(proc, mode="L").save(out_dir / "preprocessed_84x84.png")

    # Quick stack check
    stacker = FrameStacker(stack_size=4)
    obs0 = stacker.reset(proc)
    obs1 = stacker.step(proc)
    print(f"Stack shapes: reset->{obs0.shape}, step->{obs1.shape} (expected (4,84,84))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

