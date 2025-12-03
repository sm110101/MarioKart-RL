import sys
from pathlib import Path

try:
    from desmume.emulator import DeSmuME
except Exception as import_error:
    print("Failed to import py-desmume (desmume.emulator). Ensure py-desmume is installed.", file=sys.stderr)
    raise import_error


def resolve_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = repo_root / "ROM" / "mariokart.nds"
    in_state = repo_root / "ROM" / "yoshi_falls_time_trial.dsv"
    out_state = repo_root / "ROM" / "yoshi_falls_time_trial_t+480.dsv"
    return rom_path, in_state, out_state


def main() -> int:
    rom_path, in_state, out_state = resolve_paths()

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not in_state.exists():
        print(f"Input savestate not found: {in_state}", file=sys.stderr)
        return 1

    frames = 480  # 8 seconds at 60 fps
    print(f"[make_savestate_offset] Loading ROM and savestate, advancing {frames} frames...")

    emu = DeSmuME()
    emu.open(str(rom_path), auto_resume=False)
    emu.savestate.load_file(str(in_state))
    emu.resume()

    for _ in range(frames):
        emu.cycle(with_joystick=True)

    # Save new savestate
    emu.pause()
    emu.savestate.save_file(str(out_state))
    print(f"[make_savestate_offset] Saved: {out_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


