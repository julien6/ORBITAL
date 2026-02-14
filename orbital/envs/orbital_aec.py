from __future__ import annotations

from typing import Any

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.dynamics import OrbitalCore
from orbital.envs.core.spaces import build_action_space, build_observation_space
from orbital.envs.rendering.pygame_renderer import PygameRenderer


class OrbitalAECEnv(AECEnv):
    metadata = {"name": "orbital_aec_v0", "render_modes": ["human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.config = OrbitalConfig(**kwargs)
        self.possible_agents = [f"sat_{i}" for i in range(self.config.num_satellites)]
        self.agents = []
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self._action_space = build_action_space()
        self._observation_space = build_observation_space()
        self.render_mode = self.config.render_mode
        self.renderer = PygameRenderer() if self.render_mode is not None else None
        self.core = OrbitalCore(self.config)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = self.possible_agents[:]
        self.core.reset(seed)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: self.core._build_info(i) for i, a in enumerate(self.agents)}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._pending_actions = {}

    def observe(self, agent):
        return self.core.observe(self.agent_name_mapping[agent])

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._pending_actions[agent] = int(action)

        if self._agent_selector.is_last():
            rewards, terms, truncs, infos = self.core.step(self._pending_actions, self.agents)
            self.rewards = rewards
            self.terminations = terms
            self.truncations = truncs
            self.infos = infos
            self._pending_actions = {}
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        if self.render_mode is None:
            return None
        return self.renderer.render(self.core, self.render_mode, show_links=self.config.show_links)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def env(**kwargs: Any):
    environment = OrbitalAECEnv(**kwargs)
    environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment
