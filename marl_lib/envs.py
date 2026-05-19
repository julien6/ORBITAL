from __future__ import annotations

from importlib import import_module
from typing import Any

from mma import MMAWrapper, Organization
import mma.presets.orbital  # noqa: F401 - registers ORBITAL MMA rules.


def make_env(env_id: str, env_kwargs: dict[str, Any], organization_path: str | None = None, mma_mode: str = "train", seed: int | None = None):
    module_name, factory_name = env_id.rsplit(".", 1)
    factory = getattr(import_module(module_name), factory_name)
    env = factory(**env_kwargs)
    if organization_path is not None:
        env = MMAWrapper(env, Organization.from_json(organization_path), mode=mma_mode, seed=seed)
    return env
