from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


@dataclass
class MAPPOConfig:
    env_id: str = "orbital.parallel_env"
    env_kwargs: dict[str, Any] = field(default_factory=lambda: {"num_satellites": 6, "max_steps": 128})
    organization_path: str | None = None
    run_dir: str = "runs/orbital_mappo"
    seed: int = 7
    total_steps: int = 4096
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    hidden_size: int = 128
    checkpoint_interval: int = 2048
    eval_interval: int = 2048
    eval_episodes: int = 4
    eval_on_update: bool = True
    target_reward: float | None = None
    patience_evals: int | None = None
    device: str = "auto"
    image_downsample: int = 1
    image_grayscale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MAPPOConfig":
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "MAPPOConfig":
        text = Path(path).read_text(encoding="utf-8")
        if yaml is None:
            import json

            return cls.from_dict(json.loads(text))
        return cls.from_dict(yaml.safe_load(text))
