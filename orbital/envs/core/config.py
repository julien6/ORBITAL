from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_REWARD_WEIGHTS = {
    "task": 1.0,
    "delivery": 1.5,
    "energy": 0.05,
    "isolation": 0.3,
    "failure": 1.0,
    "cyber": 0.4,
}

DEFAULT_ENERGY_COSTS = {
    "observe": 1.5,
    "relay": 1.0,
    "move": 1.2,
    "lowpower": 0.2,
    "cyberscan": 0.8,
    "idle": 0.4,
}


@dataclass
class OrbitalConfig:
    num_satellites: int = 6
    grid_size: int = 12
    num_tasks: int = 8
    task_spawn_rate: float = 0.15
    task_priority_mode: str = "dynamic"
    energy_budget: float = 40.0
    energy_costs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ENERGY_COSTS))
    enable_recharge: bool = True
    recharge_rate: float = 0.4
    comm_radius: int = 3
    p_link_drop: float = 0.05
    adversarial_rate: float = 0.05
    compromise_duration: int = 8
    spoof_mode: str = "obs_spoof"
    reward_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    reward_mode: str = "shared"
    max_steps: int = 256
    sunlight_period: int = 20
    render_mode: str | None = None
    show_links: bool = True

    def __post_init__(self) -> None:
        if self.num_satellites < 1:
            raise ValueError("num_satellites must be >= 1")
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")
        if not 0.0 <= self.task_spawn_rate <= 1.0:
            raise ValueError("task_spawn_rate must be in [0,1]")
        if not 0.0 <= self.p_link_drop <= 1.0:
            raise ValueError("p_link_drop must be in [0,1]")
        if not 0.0 <= self.adversarial_rate <= 1.0:
            raise ValueError("adversarial_rate must be in [0,1]")
        if self.compromise_duration < 1:
            raise ValueError("compromise_duration must be >=1")
        if self.reward_mode not in {"shared", "local"}:
            raise ValueError("reward_mode must be shared or local")
