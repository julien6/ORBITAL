from __future__ import annotations

from typing import Any

from pettingzoo.utils import wrappers

from orbital.envs.orbital_aec import OrbitalAECEnv


class Orbital3DAECEnv(OrbitalAECEnv):
    """AEC ORBITAL wrapper with 3D orbital state enabled by default."""

    metadata = {"name": "orbital3d_aec_v0", "render_modes": [
        "human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs.setdefault("world_dim", 3)
        kwargs.setdefault("render_projection", "3d")
        super().__init__(**kwargs)


def env(**kwargs: Any):
    environment = Orbital3DAECEnv(**kwargs)
    environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment
