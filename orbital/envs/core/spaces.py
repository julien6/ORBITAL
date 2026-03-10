from __future__ import annotations

from gymnasium import spaces
import numpy as np


ACTION_MAP = {
    0: "observe",
    1: "relay",
    2: "orbit_down",
    3: "orbit_up",
    4: "lowpower",
    5: "cyberscan",
    6: "idle",
}


def build_action_space() -> spaces.Discrete:
    """Return fixed discrete action space used across all ORBITAL variants."""
    return spaces.Discrete(len(ACTION_MAP))


def build_observation_space() -> spaces.Box:
    # Fixed-size vector only (MARL-friendly).
    low = np.full((16,), -1.0, dtype=np.float32)
    high = np.full((16,), 1.0, dtype=np.float32)
    high[0:7] = 1.0
    return spaces.Box(low=low, high=high, shape=(16,), dtype=np.float32)
