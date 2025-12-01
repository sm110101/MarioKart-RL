from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Ensure project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from src.utils.state_preprocessing import vector_size


class TrajectoryDataset(Dataset):
	def __init__(self, states: np.ndarray, actions: np.ndarray):
		if states.ndim != 2:
			raise ValueError("states must be 2D array (N, obs_dim)")
		if actions.ndim != 1:
			raise ValueError("actions must be 1D array (N,)")
		if states.shape[0] != actions.shape[0]:
			raise ValueError("states and actions must have same length")
		self.states = states.astype(np.float32, copy=False)
		self.actions = actions.astype(np.int64, copy=False)

	def __len__(self) -> int:
		return self.states.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		s = torch.from_numpy(self.states[idx])
		a = torch.tensor(self.actions[idx], dtype=torch.long)
		return s, a


class PolicyMLP(nn.Module):
	def __init__(self, obs_dim: int, num_actions: int, hidden_sizes=(256, 256)):
		super().__init__()
		layers = []
		last = obs_dim
		for h in hidden_sizes:
			layers.append(nn.Linear(last, h))
			layers.append(nn.ReLU())
			last = h
		layers.append(nn.Linear(last, num_actions))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@dataclass
class TrainConfig:
	data_path: str
	output_path: str = "checkpoints/bc_model.pt"
	epochs: int = 10
	batch_size: int = 256
	lr: float = 3e-4
	val_split: float = 0.1
	num_actions: int = 6
	device: str = "auto"  # "auto" | "cpu" | "cuda"


def resolve_device(pref: str) -> torch.device:
	if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
		return torch.device("cuda")
	return torch.device("cpu")


def load_dataset_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Dataset not found: {path}")
	data = np.load(path)
	if "states" not in data or "actions" not in data:
		raise ValueError("Dataset .npz must contain 'states' (N, obs_dim) and 'actions' (N,) arrays")
	return data["states"], data["actions"]


def train_bc(cfg: TrainConfig) -> None:
	os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
	states, actions = load_dataset_npz(cfg.data_path)

	obs_dim = states.shape[1]
	expected_dim = vector_size()
	if obs_dim != expected_dim:
		print(f"Warning: dataset obs_dim={obs_dim} differs from env vector_size={expected_dim}")

	full_ds = TrajectoryDataset(states, actions)
	val_len = int(len(full_ds) * cfg.val_split)
	train_len = len(full_ds) - val_len
	train_ds, val_ds = random_split(full_ds, [train_len, val_len]) if val_len > 0 else (full_ds, None)

	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None else None

	device = resolve_device(cfg.device)
	policy = PolicyMLP(obs_dim=obs_dim, num_actions=cfg.num_actions).to(device)
	optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(cfg.epochs):
		policy.train()
		total_loss = 0.0
		total_items = 0
		for batch_states, batch_actions in train_loader:
			batch_states = batch_states.to(device)
			batch_actions = batch_actions.to(device)
			logits = policy(batch_states)
			loss = criterion(logits, batch_actions)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += float(loss.item()) * batch_states.size(0)
			total_items += batch_states.size(0)
		avg_train_loss = total_loss / max(1, total_items)

		avg_val_loss = None
		if val_loader is not None:
			policy.eval()
			val_total = 0.0
			val_items = 0
			with torch.no_grad():
				for vs, va in val_loader:
					vs = vs.to(device)
					va = va.to(device)
					logits = policy(vs)
					loss = criterion(logits, va)
					val_total += float(loss.item()) * vs.size(0)
					val_items += vs.size(0)
			avg_val_loss = val_total / max(1, val_items)

		if avg_val_loss is not None:
			print(f"Epoch {epoch+1}/{cfg.epochs} - train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
		else:
			print(f"Epoch {epoch+1}/{cfg.epochs} - train_loss={avg_train_loss:.4f}")

	# Save model weights
	torch.save({"state_dict": policy.state_dict(), "obs_dim": obs_dim, "num_actions": cfg.num_actions}, cfg.output_path)
	print(f"Saved BC model to {cfg.output_path}")


def parse_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Behavior Cloning training (pure PyTorch).")
	parser.add_argument("--data", required=True, help="Path to dataset .npz with 'states' and 'actions'")
	parser.add_argument("--out", default="checkpoints/bc_model.pt", help="Output path for saved model")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--val_split", type=float, default=0.1)
	parser.add_argument("--num_actions", type=int, default=6)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	args = parser.parse_args()
	return TrainConfig(
		data_path=args.data,
		output_path=args.out,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		val_split=args.val_split,
		num_actions=args.num_actions,
		device=args.device,
	)


if __name__ == "__main__":
	cfg = parse_args()
	train_bc(cfg)


