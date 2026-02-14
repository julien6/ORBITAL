from __future__ import annotations

from typing import Any

from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.dynamics import OrbitalCore
from orbital.envs.core.spaces import build_action_space, build_observation_space
from orbital.envs.rendering.pygame_renderer import PygameRenderer


class OrbitalParallelEnv(ParallelEnv):
    metadata = {"name": "orbital_parallel_v0", "render_modes": ["human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, **kwargs: Any):
        self.config = OrbitalConfig(**kwargs)
        self.possible_agents = [f"sat_{i}" for i in range(self.config.num_satellites)]
        self.agents = self.possible_agents[:]
        self._action_space = build_action_space()
        self._observation_space = build_observation_space()
        self.core = OrbitalCore(self.config)
        self.render_mode = self.config.render_mode
        self.renderer = PygameRenderer() if self.render_mode is not None else None
        self.np_random = None

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.np_random, _ = seeding.np_random(seed)
        self.core.reset(seed)
        self.agents = self.possible_agents[:]
        obs = {a: self.core.observe(i) for i, a in enumerate(self.agents)}
        infos = {a: self.core._build_info(i) for i, a in enumerate(self.agents)}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        rewards, terms, truncs, infos = self.core.step(actions, self.agents)
        obs = {a: self.core.observe(i) for i, a in enumerate(self.agents)}
        if any(terms.values()) or any(truncs.values()):
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def render(self):
        if self.render_mode is None:
            return None
        return self.renderer.render(self.core, self.render_mode, show_links=self.config.show_links)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def parallel_env(**kwargs: Any) -> OrbitalParallelEnv:
    return OrbitalParallelEnv(**kwargs)
