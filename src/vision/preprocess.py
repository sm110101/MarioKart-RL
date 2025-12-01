from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import numpy as np


IMG_H: int = 84
IMG_W: int = 84


def preprocess_frame(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to an Atari-style observation:
    - Crop top Nintendo DS screen (first 192 rows of a stacked (384,256,3) buffer)
      If the input is already a single screen (192,256,3), this is a no-op crop.
    - Grayscale
    - Resize to (84,84)
    - Return dtype uint8

    Args:
        rgb_image: np.ndarray of shape (H, W, 3), dtype=uint8

    Returns:
        np.ndarray of shape (84, 84), dtype=uint8
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb_image.shape}")

    height, width, _ = rgb_image.shape

    # If stacked screens are provided (384x256), take the top 192 rows
    if height >= 384 and width == 256:
        top = rgb_image[:192, :, :]
    elif height >= 192 and width == 256:
        top = rgb_image[:192, :, :]
    else:
        # Fallback: assume already a single screen; do not crop
        top = rgb_image

    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


class FrameStacker:
    """
    Maintains a sliding window of the last N preprocessed frames.
    Produces an observation with shape (stack_size, 84, 84) in uint8.
    """

    def __init__(self, stack_size: int = 4) -> None:
        if stack_size < 1:
            raise ValueError("stack_size must be >= 1")
        self.stack_size: int = stack_size
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        """
        Initialize the stack by repeating the first frame.
        Expects a single preprocessed frame of shape (84,84), dtype=uint8.
        """
        if first_frame.shape != (IMG_H, IMG_W):
            raise ValueError(f"first_frame must be (84,84), got {first_frame.shape}")
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)
        return np.stack(list(self.frames), axis=0)

    def step(self, new_frame: np.ndarray) -> np.ndarray:
        """
        Append a new preprocessed frame and return the stacked observation.
        Expects a single preprocessed frame of shape (84,84), dtype=uint8.
        """
        if new_frame.shape != (IMG_H, IMG_W):
            raise ValueError(f"new_frame must be (84,84), got {new_frame.shape}")
        self.frames.append(new_frame)
        return np.stack(list(self.frames), axis=0)


