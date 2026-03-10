from __future__ import annotations

from dataclasses import dataclass, field
import math


DEFAULT_REWARD_WEIGHTS = {
    "task": 1.0,
    "delivery": 1.5,
    "energy": 0.05,
    "isolation": 0.3,
    "failure": 1.0,
    "cyber": 0.4,
    "debris_risk": 0.35,
    "collision": 2.5,
}

DEFAULT_ENERGY_COSTS = {
    "observe": 1.5,
    "relay": 1.0,
    "orbit_down": 1.0,
    "orbit_up": 1.3,
    "lowpower": 0.2,
    "cyberscan": 0.8,
    "idle": 0.4,
}


@dataclass
class OrbitalConfig:
    """Configuration for ORBITAL dynamics.

    The environment intentionally separates task servicing (`observe`) from
    mission-value realization (`relay`), so configuration jointly controls
    sensing opportunity, delivery opportunity, and resilience stressors.
    """

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
    enable_debris: bool = True
    num_debris_clouds: int = 4
    debris_spawn_rate: float = 0.10
    debris_decay: float = 0.02
    debris_drift_std: float = 0.03
    debris_spread_min: float = 0.30
    debris_spread_max: float = 0.90
    debris_risk_gain: float = 0.85
    pc_alert_threshold: float = 0.35
    pc_collision_scale: float = 0.08
    debris_mitigation_factor: float = 0.45
    reward_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    reward_mode: str = "shared"
    max_steps: int = 256
    sunlight_period: int = 20
    orbit_min_radius: float = 2.0
    orbit_max_radius: float = 8.0
    kepler_constant: float = 1.0
    orbit_shift_step: float = 0.45
    earth_radius: float = 1.0
    ground_theta: float = -math.pi / 2.0
    ground_station_thetas: tuple[float, ...] = ()
    ground_station_phis: tuple[float, ...] = ()
    ground_contact_angle: float = 0.30
    world_dim: int = 2
    inclination_max: float = math.pi / 6.0
    render_projection: str = "2d"
    render_quality: str = "medium"
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
        if self.num_debris_clouds < 0:
            raise ValueError("num_debris_clouds must be >= 0")
        if not 0.0 <= self.debris_spawn_rate <= 1.0:
            raise ValueError("debris_spawn_rate must be in [0,1]")
        if not 0.0 <= self.debris_decay <= 1.0:
            raise ValueError("debris_decay must be in [0,1]")
        if self.debris_drift_std < 0.0:
            raise ValueError("debris_drift_std must be >= 0")
        if self.debris_spread_min <= 0.0:
            raise ValueError("debris_spread_min must be > 0")
        if self.debris_spread_max < self.debris_spread_min:
            raise ValueError("debris_spread_max must be >= debris_spread_min")
        if self.debris_risk_gain < 0.0:
            raise ValueError("debris_risk_gain must be >= 0")
        if not 0.0 <= self.pc_alert_threshold <= 1.0:
            raise ValueError("pc_alert_threshold must be in [0,1]")
        if not 0.0 <= self.pc_collision_scale <= 1.0:
            raise ValueError("pc_collision_scale must be in [0,1]")
        if not 0.0 <= self.debris_mitigation_factor <= 1.0:
            raise ValueError("debris_mitigation_factor must be in [0,1]")
        if self.reward_mode not in {"shared", "local"}:
            raise ValueError("reward_mode must be shared or local")
        if self.orbit_min_radius <= 0.0:
            raise ValueError("orbit_min_radius must be > 0")
        if self.orbit_max_radius <= self.orbit_min_radius:
            raise ValueError("orbit_max_radius must be > orbit_min_radius")
        if self.earth_radius <= 0.0:
            raise ValueError("earth_radius must be > 0")
        if self.orbit_min_radius <= self.earth_radius:
            raise ValueError("orbit_min_radius must be > earth_radius")
        if self.kepler_constant <= 0.0:
            raise ValueError("kepler_constant must be > 0")
        if self.orbit_shift_step <= 0.0:
            raise ValueError("orbit_shift_step must be > 0")
        if not 0.0 < self.ground_contact_angle <= math.pi:
            raise ValueError("ground_contact_angle must be in (0, pi]")
        if self.world_dim not in {2, 3}:
            raise ValueError("world_dim must be 2 or 3")
        if not 0.0 <= self.inclination_max <= (math.pi / 2.0):
            raise ValueError("inclination_max must be in [0, pi/2]")
        if self.render_projection not in {"2d", "3d"}:
            raise ValueError("render_projection must be '2d' or '3d'")
        if self.render_quality not in {"ultra_low", "low", "medium", "high"}:
            raise ValueError("render_quality must be 'ultra_low', 'low', 'medium', or 'high'")
        if len(self.ground_station_thetas) == 0:
            self.ground_station_thetas = (self.ground_theta, self.ground_theta + math.pi)
        if len(self.ground_station_thetas) < 1:
            raise ValueError("ground_station_thetas must contain at least one station angle")
        if len(self.ground_station_phis) == 0:
            self.ground_station_phis = tuple(0.0 for _ in self.ground_station_thetas)
        if len(self.ground_station_phis) != len(self.ground_station_thetas):
            raise ValueError("ground_station_phis must match ground_station_thetas length")
