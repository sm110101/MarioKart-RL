from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any
import yaml
import numpy as np
import torch
import torch.nn as nn

# Ensure project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from src.envs.mario_kart_yoshi_falls_env import MarioKartYoshiFallsEnv
from src.utils.state_preprocessing import vector_size


class ActorCritic(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
		super().__init__()
		def mlp(in_dim, layers):
			m = []
			last = in_dim
			for h in layers:
				m += [nn.Linear(last, h), nn.ReLU()]
				last = h
			return nn.Sequential(*m)
		self.pi_body = mlp(obs_dim, hidden)
		self.v_body = mlp(obs_dim, hidden)
		self.pi_head = nn.Linear(hidden[-1], act_dim)
		self.v_head = nn.Linear(hidden[-1], 1)

	def forward(self, x: torch.Tensor):
		pi = self.pi_head(self.pi_body(x))
		v = self.v_head(self.v_body(x)).squeeze(-1)
		return pi, v


def load_env_config(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main():
	parser = argparse.ArgumentParser(description="Play PPO-lite policy (PyTorch) with live viewing.")
	parser.add_argument("--config", default="configs/env.yaml", help="Path to env.yaml")
	parser.add_argument("--checkpoint", default="checkpoints/ppo_lite_latest.pt", help="Path to PPO checkpoint")
	parser.add_argument("--episodes", type=int, default=2)
	parser.add_argument("--render", action="store_true", help="Open emulator window")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--sleep_seconds", type=float, default=None, help="Override per-cycle sleep for playback (e.g., 0.01)")
	args = parser.parse_args()

	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

	ckpt = torch.load(args.checkpoint, map_location="cpu")
	obs_dim = int(ckpt.get("obs_dim", vector_size()))
	act_dim = int(ckpt.get("act_dim", 6))

	device = torch.device("cuda" if (args.device in ["cuda", "auto"] and torch.cuda.is_available()) else "cpu")
	policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
	policy.load_state_dict(ckpt["state_dict"])
	policy.to(device)
	policy.eval()

	cfg = load_env_config(args.config)
	# Ensure normal speed for playback: use override if provided, else config value
	if args.sleep_seconds is not None:
		cfg["per_cycle_sleep_seconds"] = float(args.sleep_seconds)

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
				to = torch.from_numpy(obs).unsqueeze(0).to(device)
				logits, _ = policy(to)
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


