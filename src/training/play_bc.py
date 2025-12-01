from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any
import yaml
import numpy as np
import torch

# Ensure project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from src.envs.mario_kart_yoshi_falls_env import MarioKartYoshiFallsEnv
from src.training.train_bc import PolicyMLP  # reuse the same architecture


def load_env_config(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main():
	parser = argparse.ArgumentParser(description="Play using a trained BC model (PyTorch).")
	parser.add_argument("--config", default="configs/env.yaml", help="Path to env.yaml")
	parser.add_argument("--checkpoint", default="checkpoints/bc_model.pt", help="Path to BC checkpoint")
	parser.add_argument("--episodes", type=int, default=2)
	parser.add_argument("--render", action="store_true", help="Open emulator window")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	args = parser.parse_args()

	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

	ckpt = torch.load(args.checkpoint, map_location="cpu")
	obs_dim = int(ckpt.get("obs_dim", 3))
	num_actions = int(ckpt.get("num_actions", 6))

	device = torch.device("cuda" if (args.device in ["cuda", "auto"] and torch.cuda.is_available()) else "cpu")

	policy = PolicyMLP(obs_dim=obs_dim, num_actions=num_actions)
	policy.load_state_dict(ckpt["state_dict"])
	policy.to(device)
	policy.eval()

	cfg = load_env_config(args.config)
	env = MarioKartYoshiFallsEnv(cfg)

	window = None
	if args.render:
		try:
			window = env.emu.emu.create_sdl_window()  # type: ignore[attr-defined]
		except Exception:
			window = None

	for ep in range(args.episodes):
		obs, info = env.reset()
		done = False
		trunc = False
		steps = 0
		ret = 0.0
		while not (done or trunc):
			with torch.no_grad():
				tobs = torch.from_numpy(obs).unsqueeze(0).to(device)
				logits = policy(tobs)
				action = int(torch.argmax(logits, dim=-1).item())
			obs, rew, done, trunc, info = env.step(action)
			ret += float(rew)
			steps += 1
			if window is not None:
				try:
					window.process_input()
					window.draw()
				except Exception:
					pass
		print(f"Episode {ep+1}: steps={steps}, reward_sum={ret:.3f}")


if __name__ == "__main__":
	main()


