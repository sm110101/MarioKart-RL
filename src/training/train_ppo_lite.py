from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

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

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		pi = self.pi_head(self.pi_body(x))
		v = self.v_head(self.v_body(x)).squeeze(-1)
		return pi, v


@dataclass
class PPOConfigLite:
	env_config_path: str = "configs/env.yaml"
	iterations: int = 10
	steps_per_iter: int = 4096
	minibatch_size: int = 256
	epochs: int = 4
	gamma: float = 0.99
	lambda_: float = 0.95
	clip_ratio: float = 0.2
	lr: float = 3e-4
	device: str = "auto"
	fast_sleep_seconds: float = 0.0  # speed during training


def load_env_cfg(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def make_env(env_cfg: Dict[str, Any], fast_sleep_seconds: float) -> MarioKartYoshiFallsEnv:
	override = dict(env_cfg)
	override["per_cycle_sleep_seconds"] = float(fast_sleep_seconds)
	return MarioKartYoshiFallsEnv(override)


def ppo_update(model, optimizer, obs, act, ret, adv, old_logp, cfg: PPOConfigLite, device):
	model.train()
	n = obs.size(0)
	idx = torch.randperm(n)
	for _ in range(cfg.epochs):
		for start in range(0, n, cfg.minibatch_size):
			sel = idx[start:start+cfg.minibatch_size]
			b_obs = obs[sel].to(device)
			b_act = act[sel].to(device)
			b_ret = ret[sel].to(device)
			b_adv = adv[sel].to(device)
			b_old_logp = old_logp[sel].to(device)

			logits, v = model(b_obs)
			dist = torch.distributions.Categorical(logits=logits)
			logp = dist.log_prob(b_act)
			ratio = torch.exp(logp - b_old_logp)
			clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * b_adv
			policy_loss = -(torch.min(ratio * b_adv, clip_adv)).mean()
			value_loss = 0.5 * (b_ret - v).pow(2).mean()
			entropy = dist.entropy().mean()
			loss = policy_loss + value_loss * 0.5 - 0.01 * entropy

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


def collect_trajectory(env: MarioKartYoshiFallsEnv, steps: int, model, device, gamma: float, lambda_: float, window=None, render_every: int = 1):
	obs_dim = vector_size()
	obs_buf = []
	act_buf = []
	reward_buf = []
	done_buf = []
	value_buf = []
	logp_buf = []

	obs, info = env.reset()
	ep_ret = 0.0
	ep_len = 0
	for t in range(steps):
		with torch.no_grad():
			to = torch.from_numpy(obs).unsqueeze(0).to(device)
			logits, v = model(to)
			dist = torch.distributions.Categorical(logits=logits)
			action = int(dist.sample().item())
			logp = dist.log_prob(torch.tensor([action], device=device)).squeeze(0)
		next_obs, reward, done, truncated, info = env.step(action)

		obs_buf.append(obs.copy())
		act_buf.append(action)
		reward_buf.append(float(reward))
		done_buf.append(bool(done or truncated))
		value_buf.append(float(v.item()))
		logp_buf.append(float(logp.item()))

		ep_ret += float(reward)
		ep_len += 1
		obs = next_obs
		# Live rendering during training if requested
		if window is not None and (t % max(1, int(render_every)) == 0):
			try:
				window.process_input()
				window.draw()
			except Exception:
				pass
		if done or truncated:
			obs, info = env.reset()
			ep_ret = 0.0
			ep_len = 0

	# Compute GAE-lambda
	with torch.no_grad():
		to = torch.from_numpy(obs).unsqueeze(0).to(device)
		_, last_v = model(to)
		last_v = float(last_v.item())
	values = np.array(value_buf + [last_v], dtype=np.float32)
	rewards = np.array(reward_buf, dtype=np.float32)
	dones = np.array(done_buf, dtype=np.bool_)

	adv = np.zeros_like(rewards, dtype=np.float32)
	gae = 0.0
	for t in reversed(range(steps)):
		nonterminal = 1.0 - float(dones[t])
		delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
		gae = delta + gamma * lambda_ * nonterminal * gae
		adv[t] = gae
	ret = adv + values[:-1]

	return (
		torch.tensor(np.array(obs_buf), dtype=torch.float32),
		torch.tensor(np.array(act_buf), dtype=torch.int64),
		torch.tensor(ret, dtype=torch.float32),
		torch.tensor(adv, dtype=torch.float32),
		torch.tensor(np.array(logp_buf), dtype=torch.float32),
	)


def main():
	parser = argparse.ArgumentParser(description="PPO-lite training (pure PyTorch).")
	parser.add_argument("--env_config", default="configs/env.yaml")
	parser.add_argument("--iterations", type=int, default=10)
	parser.add_argument("--steps_per_iter", type=int, default=4096)
	parser.add_argument("--minibatch_size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--lambda_", type=float, default=0.95)
	parser.add_argument("--clip_ratio", type=float, default=0.2)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--fast_sleep_seconds", type=float, default=0.0, help="Set 0.0 for max speed during training")
	parser.add_argument("--render", action="store_true", help="Open emulator window to watch training live")
	parser.add_argument("--render_every", type=int, default=1, help="Draw every N environment steps (>=1)")
	args = parser.parse_args()

	cfg = PPOConfigLite(
		env_config_path=args.env_config,
		iterations=args.iterations,
		steps_per_iter=args.steps_per_iter,
		minibatch_size=args.minibatch_size,
		epochs=args.epochs,
		gamma=args.gamma,
		lambda_=args.lambda_,
		clip_ratio=args.clip_ratio,
		lr=args.lr,
		device=args.device,
		fast_sleep_seconds=args.fast_sleep_seconds,
	)

	env_cfg = load_env_cfg(cfg.env_config_path)
	env = make_env(env_cfg, cfg.fast_sleep_seconds)

	# Optional live viewing
	window = None
	if args.render:
		try:
			window = env.emu.emu.create_sdl_window()  # type: ignore[attr-defined]
		except Exception:
			window = None

	device = torch.device("cuda" if (cfg.device in ["cuda", "auto"] and torch.cuda.is_available()) else "cpu")
	model = ActorCritic(obs_dim=vector_size(), act_dim=6).to(device)
	optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

	for it in range(cfg.iterations):
		obs, act, ret, adv, logp = collect_trajectory(
			env, cfg.steps_per_iter, model, device, cfg.gamma, cfg.lambda_, window=window, render_every=max(1, args.render_every)
		)
		# Normalize advantages
		adv = (adv - adv.mean()) / (adv.std() + 1e-8)
		ppo_update(model, optimizer, obs, act, ret, adv, logp, cfg, device)
		print(f"Iter {it+1}/{cfg.iterations} - done")
		# Save checkpoint
		try:
			os.makedirs("checkpoints", exist_ok=True)
			payload = {
				"state_dict": model.state_dict(),
				"obs_dim": vector_size(),
				"act_dim": 6,
				"iter": it + 1,
				"saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
			}
			ckpt_path = os.path.join("checkpoints", f"ppo_lite_iter_{it+1}.pt")
			torch.save(payload, ckpt_path)
			# Also update latest
			torch.save(payload, os.path.join("checkpoints", "ppo_lite_latest.pt"))
		except Exception as e:
			print(f"Warning: failed to save checkpoint at iter {it+1}: {e}")


if __name__ == "__main__":
	main()


