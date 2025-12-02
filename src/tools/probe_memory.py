import sys
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from desmume.emulator import DeSmuME
except Exception as import_error:
    print("Failed to import py-desmume (desmume.emulator). Ensure py-desmume is installed.", file=sys.stderr)
    raise import_error


def resolve_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    rom_path = repo_root / "ROM" / "mariokart.nds"
    savestate_path = repo_root / "ROM" / "yoshi_falls_time_trial_t+420.dsv"
    mem_cfg_path = repo_root / "src" / "configs" / "memory_addresses.yaml"
    return rom_path, savestate_path, mem_cfg_path


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"Memory config not found at {path}", file=sys.stderr)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_value(memory, addr: int, size: int, little_endian: bool, signed: bool) -> int:
    """
    Read an integer value using the documented API:
      DeSmuME_Memory.read(start, end, size, signed)
    When start == end, an integer is returned (see docs).
    """
    native_read = getattr(memory, "read", None)
    if not callable(native_read):
        raise RuntimeError("DeSmuME_Memory.read is not available on this binding")
    return int(native_read(addr, addr, size, signed))


def main() -> int:
    rom_path, savestate_path, mem_cfg_path = resolve_paths()

    if not rom_path.exists():
        print(f"ROM not found: {rom_path}", file=sys.stderr)
        return 1
    if not savestate_path.exists():
        print(f"Savestate not found: {savestate_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(mem_cfg_path)
    if not cfg:
        print("Empty memory config; fill in src/configs/memory_addresses.yaml", file=sys.stderr)
        return 1

    emu = DeSmuME()
    emu.open(str(rom_path), auto_resume=False)
    emu.savestate.load_file(str(savestate_path))
    emu.resume()

    names = ["progress", "speed", "wrong_way", "lap"]
    print("Stepping 720 frames and reading memory each frame...")
    for frame in range(720):
        emu.cycle(with_joystick=True)
        row = {"frame": frame}
        for name in names:
            entry = cfg.get(name)
            if not entry:
                row[name] = None
                continue
            addr = int(entry["addr"])
            size = int(entry["size"])
            little_endian = bool(entry.get("little_endian", True))
            signed = bool(entry.get("signed", False))
            try:
                value = read_value(emu.memory, addr, size, little_endian, signed)
                scale = float(entry.get("scale", 1.0))
                row[name + "_raw"] = value
                row[name] = value * scale
            except Exception as e:
                value = f"err:{e}"
                row[name] = value
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

