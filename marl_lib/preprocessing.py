from __future__ import annotations

import numpy as np


def preprocess_observation(obs, image_downsample: int = 1, image_grayscale: bool = False) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim == 3:
        if image_downsample > 1:
            arr = arr[::image_downsample, ::image_downsample]
        if image_grayscale and arr.shape[-1] > 1:
            arr = arr.astype(np.float32).mean(axis=-1)
    arr = arr.astype(np.float32).reshape(-1)
    if np.asarray(obs).dtype == np.uint8:
        arr = arr / 255.0
    return arr
