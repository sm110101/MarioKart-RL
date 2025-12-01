from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any
import yaml
import numpy as np

# Ensure project root is on sys.path when running as a script
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from src.envs.mario_kart_yoshi_falls_env import MarioKartYoshiFallsEnv


def load_env_config(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main():
	parser = argparse.ArgumentParser(description="Evaluate (random policy) with optional live viewing.")
	parser.add_argument("--config", default="configs/env.yaml", help="Path to env.yaml")
	parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
	parser.add_argument("--render", action="store_true", help="Open the emulator window for live viewing")
	args = parser.parse_args()

	cfg = load_env_config(args.config)
	env = MarioKartYoshiFallsEnv(cfg)

	# Optional live window: create via the underlying DeSmuME instance
	window = None
	if args.render:
		try:
			window = env.emu.emu.create_sdl_window()  # type: ignore[attr-defined]
		except Exception:
			window = None

	total_rewards = []
	for ep in range(args.episodes):
		obs, info = env.reset()
		ep_reward = 0.0
		terminated = False
		truncated = False
		steps = 0
		while not (terminated or truncated):
			action = env.action_space.sample()
			obs, reward, terminated, truncated, info = env.step(action)
			ep_reward += float(reward)
			steps += 1
			# If rendering, keep the SDL window responsive and draw the current framebuffer
			if window is not None:
				try:
					window.process_input()
					window.draw()
				except Exception:
					pass
			# If rendering, the emulator window is drawn in its own loop inside SDL; we just continue stepping
		total_rewards.append(ep_reward)
		print(f"Episode {ep+1}: steps={steps}, reward_sum={ep_reward:.3f}")

	if total_rewards:
		print(f"Avg reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.3f}")


if __name__ == "__main__":
	main()


