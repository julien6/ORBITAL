from __future__ import annotations

from gymnasium import spaces
import numpy as np


ACTION_MAP = {
    0: "observe",
    1: "relay",
    2: "move",
    3: "lowpower",
    4: "cyberscan",
    5: "idle",
}


def build_action_space() -> spaces.Discrete:
    return spaces.Discrete(len(ACTION_MAP))


def build_observation_space() -> spaces.Box:
    # Fixed-size vector only (MARL-friendly).
    low = np.full((14,), -1.0, dtype=np.float32)
    high = np.full((14,), 1.0, dtype=np.float32)
    high[0:7] = 1.0
    return spaces.Box(low=low, high=high, shape=(14,), dtype=np.float32)
