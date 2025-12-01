from __future__ import annotations

from typing import Dict, Any
import numpy as np

# Placeholder scales; will be calibrated later
_MAX_RAW_SPEED = 1023.0
_MAX_RAW_PROGRESS = 65535.0


def _norm_or_zero(value: int | float | None, max_val: float) -> float:
	if value is None:
		return 0.0
	try:
		v = float(value)
		if max_val > 0:
			return float(np.clip(v / max_val, 0.0, 1.0))
		return 0.0
	except Exception:
		return 0.0


def state_dict_to_vector(state: Dict[str, Any]) -> np.ndarray:
	"""
	Convert the raw state dict from MarioKartInterface into a fixed-size vector.
	Features (placeholder):
	- normalized speed (0..1)
	- normalized progress (0..1)
	- lap as scalar (float)
	"""
	speed = _norm_or_zero(state.get("speed_raw"), _MAX_RAW_SPEED)
	progress = _norm_or_zero(state.get("progress_raw"), _MAX_RAW_PROGRESS)
	lap_val = float(state.get("lap") or 0.0)
	vec = np.array([speed, progress, lap_val], dtype=np.float32)
	return vec


def vector_size() -> int:
	"""Return the length of the observation vector."""
	return 3


