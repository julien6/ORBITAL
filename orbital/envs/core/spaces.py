from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces
except ModuleNotFoundError:  # Core dynamics can be imported without PettingZoo deps.
    spaces = None


ACTION_MAP = {
    0: "observe",
    1: "relay_ground",
    2: "relay_sat",
    3: "orbit_down",
    4: "orbit_up",
    5: "lowpower",
    6: "cyberscan",
    7: "idle",
}

OBSERVATION_SIZE = 20


def build_action_space() -> spaces.Discrete:
    """Return fixed discrete action space used across all ORBITAL variants."""
    if spaces is None:
        raise ModuleNotFoundError("gymnasium is required to build ORBITAL action spaces")
    return spaces.Discrete(len(ACTION_MAP))


def build_observation_space() -> spaces.Box:
    if spaces is None:
        raise ModuleNotFoundError("gymnasium is required to build ORBITAL observation spaces")
    # Fixed-size vector only (MARL-friendly).
    low = np.full((OBSERVATION_SIZE,), -1.0, dtype=np.float32)
    high = np.full((OBSERVATION_SIZE,), 1.0, dtype=np.float32)
    high[0:OBSERVATION_SIZE] = 1.0
    return spaces.Box(low=low, high=high, shape=(OBSERVATION_SIZE,), dtype=np.float32)
