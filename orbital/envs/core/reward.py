from __future__ import annotations


def compute_shared_reward(components: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted mission value from per-step component signals."""
    return (
        weights.get("task", 0.0) * components.get("task", 0.0)
        + weights.get("delivery", 0.0) * components.get("delivery", 0.0)
        + weights.get("knowledge", 0.0) * components.get("knowledge", 0.0)
        - weights.get("energy", 0.0) * components.get("energy", 0.0)
        - weights.get("overflow", 0.0) * components.get("overflow", 0.0)
        - weights.get("data_loss", 0.0) * components.get("data_loss", 0.0)
        - weights.get("health", 0.0) * components.get("health", 0.0)
        - weights.get("isolation", 0.0) * components.get("isolation", 0.0)
        - weights.get("failure", 0.0) * components.get("failure", 0.0)
        - weights.get("cyber", 0.0) * components.get("cyber", 0.0)
        - weights.get("jam", 0.0) * components.get("jam", 0.0)
        - weights.get("forced_action", 0.0) * components.get("forced_action", 0.0)
        - weights.get("atmospheric_drag", 0.0) * components.get("atmospheric_drag", 0.0)
        - weights.get("debris_risk", 0.0) * components.get("debris_risk", 0.0)
        - weights.get("collision", 0.0) * components.get("collision", 0.0)
    )
