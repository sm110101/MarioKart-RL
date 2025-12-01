from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any, List
import yaml
import numpy as np

# Ensure project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from src.envs.mario_kart_yoshi_falls_env import MarioKartYoshiFallsEnv
from src.utils.state_preprocessing import state_dict_to_vector

try:
	import keyboard  # type: ignore
	_HAS_KBD = True
except Exception:
	_HAS_KBD = False


def load_env_config(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def infer_action_from_keys() -> int:
	"""
	Map host keys to discrete action ids:
	  0: no-op
	  1: A (accelerate)            -> x
	  2: A + LEFT                  -> x + left
	  3: A + RIGHT                 -> x + right
	  4: A + R (drift/jump)        -> x + space
	  5: release / coast (default)
	Note: 'z' (reverse) is not represented in the discrete set; if pressed without 'x', we'll pick 5.
	"""
	if not _HAS_KBD:
		return 0
	left = keyboard.is_pressed("left")
	right = keyboard.is_pressed("right")
	accel = keyboard.is_pressed("x")
	drift = keyboard.is_pressed("space")
	reverse = keyboard.is_pressed("z")

	if accel and left and not right and not drift:
		return 2
	if accel and right and not left and not drift:
		return 3
	if accel and drift:
		return 4
	if accel:
		return 1
	# If reverse only, we coast (no dedicated reverse action in this set)
	if reverse:
		return 5
	return 0


def main():
	parser = argparse.ArgumentParser(description="Record demonstrations (state-action pairs) via keyboard control.")
	parser.add_argument("--config", default="configs/env.yaml", help="Path to env.yaml")
	parser.add_argument("--out", default="data/demos_mkds.npz", help="Output .npz path")
	parser.add_argument("--steps", type=int, default=3000, help="Max total steps to record")
	parser.add_argument("--render", action="store_true", help="Open emulator window")
	args = parser.parse_args()

	if not _HAS_KBD:
		raise RuntimeError("keyboard module not available. Install with: pip install keyboard")

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

	cfg = load_env_config(args.config)
	env = MarioKartYoshiFallsEnv(cfg)

	window = None
	if args.render:
		try:
			window = env.emu.emu.create_sdl_window()  # type: ignore[attr-defined]
		except Exception:
			window = None

	states: List[np.ndarray] = []
	actions: List[int] = []

	obs, info = env.reset()
	total = 0
	print("Recording started. Use arrow keys + X/Space. Press ESC to stop.")
	try:
		while total < args.steps and not keyboard.is_pressed("esc"):
			act = infer_action_from_keys()
			obs, reward, terminated, truncated, info = env.step(act)

			# Store processed vector and action
			states.append(obs.copy())
			actions.append(int(act))
			total += 1

			if window is not None:
				try:
					window.process_input()
					window.draw()
				except Exception:
					pass

			# If episode ends (lap completed or truncated), reset and continue
			if terminated or truncated:
				obs, info = env.reset()
	except KeyboardInterrupt:
		pass

	arr_states = np.stack(states, axis=0).astype(np.float32) if states else np.zeros((0, 3), dtype=np.float32)
	arr_actions = np.array(actions, dtype=np.int64) if actions else np.zeros((0,), dtype=np.int64)
	np.savez(args.out, states=arr_states, actions=arr_actions)
	print(f"Saved {arr_states.shape[0]} samples to {args.out}")


if __name__ == "__main__":
	main()


