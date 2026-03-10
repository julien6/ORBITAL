from __future__ import annotations

from typing import Any

from orbital.envs.orbital_parallel import OrbitalParallelEnv


class Orbital3DParallelEnv(OrbitalParallelEnv):
    """Parallel ORBITAL wrapper with 3D orbital state enabled by default."""

    metadata = {"name": "orbital3d_parallel_v0", "render_modes": [
        "human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs.setdefault("world_dim", 3)
        kwargs.setdefault("render_projection", "3d")
        super().__init__(**kwargs)


def parallel_env(**kwargs: Any) -> Orbital3DParallelEnv:
    return Orbital3DParallelEnv(**kwargs)
