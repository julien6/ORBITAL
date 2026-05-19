from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from mma.organization import Organization
from mma.registry import get_goal_rule, get_role_rule


class MMAWrapper:
    """PettingZoo Parallel wrapper applying MOISE+MARL roles and goals.

    Roles correct invalid sampled actions after policy selection. Goals shape
    rewards only in training mode.
    """

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        env: Any,
        organization: Organization | None = None,
        mode: str = "train",
        history_len: int | None = None,
        seed: int | None = None,
    ):
        if mode not in {"train", "eval", "infer"}:
            raise ValueError("mode must be train, eval, or infer")
        self.env = env
        self.organization = organization
        self.mode = mode
        self.history = deque(maxlen=history_len)
        self.rng = np.random.default_rng(seed)
        self.last_obs: dict[str, Any] = {}
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = getattr(env, "possible_agents", [])
        self.agents = getattr(env, "agents", [])

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.history.clear()
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = getattr(self.env, "agents", [])
        self.last_obs = dict(obs)
        infos = self._augment_infos(infos, obs, {}, {})
        return obs, infos

    def step(self, actions: dict[str, int]):
        allowed = self.allowed_actions(self.last_obs)
        corrected: dict[str, int] = {}
        replaced: dict[str, bool] = {}
        for agent, action in actions.items():
            allowed_agent = allowed.get(agent)
            if not allowed_agent:
                allowed_agent = list(range(self.action_space(agent).n))
            if int(action) in allowed_agent:
                corrected[agent] = int(action)
                replaced[agent] = False
            else:
                corrected[agent] = int(self.rng.choice(allowed_agent))
                replaced[agent] = True

        obs, rewards, terms, truncs, infos = self.env.step(corrected)
        shaping = {agent: 0.0 for agent in rewards}
        if self.organization is not None and self.mode == "train":
            joint = {"obs": obs, "infos": infos}
            for agent in rewards:
                shaping[agent] = self._goal_bonus(agent, joint, actions, corrected, rewards, infos)
                rewards[agent] = float(rewards[agent] + shaping[agent])

        infos = self._augment_infos(infos, obs, actions, corrected, allowed, replaced, shaping)
        self.history.append(
            {
                "obs": self.last_obs,
                "next_obs": obs,
                "actions": dict(actions),
                "executed_actions": dict(corrected),
                "rewards": dict(rewards),
                "infos": infos,
            }
        )
        self.last_obs = dict(obs)
        self.agents = getattr(self.env, "agents", [])
        return obs, rewards, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def allowed_actions(self, obs: dict[str, Any] | None = None) -> dict[str, list[int]]:
        if self.organization is None:
            return {agent: list(range(self.action_space(agent).n)) for agent in getattr(self.env, "agents", [])}
        obs = self.last_obs if obs is None else obs
        joint = {"obs": obs, "infos": {}}
        out: dict[str, list[int]] = {}
        for agent in getattr(self.env, "agents", []):
            allowed: set[int] | None = None
            assignment = self.organization.assignment_for(agent)
            for role_name in assignment.roles:
                role = self.organization.roles.get(role_name)
                if role is None:
                    continue
                for rule in role.rules:
                    actions = set(int(a) for a in get_role_rule(rule.name)(list(self.history), joint, agent, rule.params))
                    allowed = actions if allowed is None else allowed & actions
            if allowed is None or not allowed:
                allowed = set(range(self.action_space(agent).n))
            out[agent] = sorted(a for a in allowed if self.action_space(agent).contains(a))
            if not out[agent]:
                out[agent] = [self.action_space(agent).n - 1]
        return out

    def _goal_bonus(
        self,
        agent: str,
        joint: dict[str, Any],
        sampled_actions: dict[str, int],
        executed_actions: dict[str, int],
        rewards: dict[str, float],
        infos: dict[str, Any],
    ) -> float:
        if self.organization is None:
            return 0.0
        total = 0.0
        assignment = self.organization.assignment_for(agent)
        for goal_name in assignment.goals:
            goal = self.organization.goals.get(goal_name)
            if goal is None:
                continue
            for rule in goal.rules:
                total += float(
                    get_goal_rule(rule.name)(
                        list(self.history),
                        joint,
                        sampled_actions,
                        executed_actions,
                        rewards,
                        infos,
                        agent,
                        rule.params,
                    )
                )
        return total

    def _augment_infos(
        self,
        infos: dict[str, Any],
        obs: dict[str, Any],
        sampled: dict[str, int],
        executed: dict[str, int],
        allowed: dict[str, list[int]] | None = None,
        replaced: dict[str, bool] | None = None,
        shaping: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        allowed = self.allowed_actions(obs) if allowed is None else allowed
        replaced = replaced or {}
        shaping = shaping or {}
        out = {agent: dict(info) for agent, info in infos.items()}
        for agent in out:
            out[agent]["mma_allowed_actions"] = list(allowed.get(agent, []))
            if agent in sampled:
                out[agent]["mma_original_action"] = int(sampled[agent])
                out[agent]["mma_executed_action"] = int(executed.get(agent, sampled[agent]))
                out[agent]["mma_action_replaced"] = bool(replaced.get(agent, False))
            out[agent]["mma_goal_shaping"] = float(shaping.get(agent, 0.0))
        return out
