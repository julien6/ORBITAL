from __future__ import annotations


def compute_shared_reward(components: dict[str, float], weights: dict[str, float]) -> float:
    return (
        weights.get("task", 0.0) * components.get("task", 0.0)
        + weights.get("delivery", 0.0) * components.get("delivery", 0.0)
        - weights.get("energy", 0.0) * components.get("energy", 0.0)
        - weights.get("isolation", 0.0) * components.get("isolation", 0.0)
        - weights.get("failure", 0.0) * components.get("failure", 0.0)
        - weights.get("cyber", 0.0) * components.get("cyber", 0.0)
    )
