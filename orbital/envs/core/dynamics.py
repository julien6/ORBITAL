from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.reward import compute_shared_reward
from orbital.envs.core.spaces import ACTION_MAP


@dataclass
class Task:
    x: int
    y: int
    priority: float
    active: bool = True
    age: int = 0


class OrbitalCore:
    def __init__(self, config: OrbitalConfig):
        self.config = config
        self.num_agents = config.num_satellites
        self.ground = np.array([0, 0], dtype=np.int32)
        self.rng = np.random.default_rng()
        self.reset(seed=None)

    def reset(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0
        gs = self.config.grid_size
        self.positions = self.rng.integers(0, gs, size=(self.num_agents, 2), endpoint=False)
        self.energy = np.full((self.num_agents,), self.config.energy_budget, dtype=np.float32)
        self.compromised_for = np.zeros((self.num_agents,), dtype=np.int32)
        self.scan_boost = np.zeros((self.num_agents,), dtype=np.int32)
        self.buffered_data = np.zeros((self.num_agents,), dtype=np.float32)
        self.tasks = [self._spawn_task() for _ in range(self.config.num_tasks)]
        self.comm_adj = np.zeros((self.num_agents, self.num_agents), dtype=np.bool_)
        self.last_reward_components = {"task": 0.0, "delivery": 0.0, "energy": 0.0, "isolation": 0.0, "failure": 0.0, "cyber": 0.0}
        self.last_reward = 0.0
        self.delivered_total = 0.0
        self.update_comm_graph()

    def _spawn_task(self) -> Task:
        gs = self.config.grid_size
        prio = float(self.rng.uniform(0.2, 1.0))
        return Task(int(self.rng.integers(0, gs)), int(self.rng.integers(0, gs)), prio, True, 0)

    def _is_alive(self, i: int) -> bool:
        return self.energy[i] > 0.0

    def update_comm_graph(self) -> None:
        n = self.num_agents
        adj = np.zeros((n, n), dtype=np.bool_)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.abs(self.positions[i] - self.positions[j]).sum()
                if dist <= self.config.comm_radius and self.rng.random() > self.config.p_link_drop:
                    adj[i, j] = True
                    adj[j, i] = True
        self.comm_adj = adj

    def _in_sunlight(self) -> bool:
        phase = self.t % self.config.sunlight_period
        return phase < (self.config.sunlight_period // 2)

    def _apply_move(self, i: int) -> None:
        delta = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.int32)
        d = delta[self.rng.integers(0, len(delta))]
        self.positions[i] = np.clip(self.positions[i] + d, 0, self.config.grid_size - 1)

    def _action_name(self, a: int) -> str:
        return ACTION_MAP.get(int(a), "idle")

    def step(self, actions: dict[str, int], agent_names: list[str]) -> tuple[dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        n = self.num_agents
        serviced = np.zeros((n,), dtype=np.float32)
        delivered = np.zeros((n,), dtype=np.float32)
        energy_spent = np.zeros((n,), dtype=np.float32)
        cyber_penalty = np.zeros((n,), dtype=np.float32)

        # adversarial event
        if self.rng.random() < self.config.adversarial_rate:
            healthy = np.where(self.compromised_for <= 0)[0]
            if len(healthy) > 0:
                idx = int(self.rng.choice(healthy))
                self.compromised_for[idx] = self.config.compromise_duration

        for i, name in enumerate(agent_names):
            act = int(actions.get(name, 5))
            if not self._is_alive(i):
                continue

            if self.compromised_for[i] > 0 and self.config.spoof_mode == "action_noise" and self.rng.random() < 0.25:
                act = int(self.rng.integers(0, 6))
                cyber_penalty[i] += 0.5

            action_name = self._action_name(act)
            cost = self.config.energy_costs.get(action_name, self.config.energy_costs["idle"])
            self.energy[i] -= cost
            energy_spent[i] += cost

            if self.energy[i] <= 0:
                self.energy[i] = 0.0
                continue

            if action_name == "move":
                self._apply_move(i)
            elif action_name == "observe":
                for task in self.tasks:
                    if not task.active:
                        continue
                    d = np.abs(self.positions[i] - np.array([task.x, task.y])).sum()
                    if d <= 1:
                        task.active = False
                        serviced[i] += task.priority
                        self.buffered_data[i] += task.priority
                        break
            elif action_name == "relay":
                if self.buffered_data[i] > 0.0:
                    at_ground = np.abs(self.positions[i] - self.ground).sum() <= 1
                    if self.compromised_for[i] > 0 and self.config.spoof_mode == "comm_jam" and self.rng.random() < 0.35:
                        cyber_penalty[i] += 1.0
                    elif at_ground or self._has_path_to_ground(i):
                        out = min(1.0, self.buffered_data[i])
                        self.buffered_data[i] -= out
                        delivered[i] += out
                        self.delivered_total += out
            elif action_name == "cyberscan":
                self.scan_boost[i] = 4
                if self.compromised_for[i] > 0 and self.rng.random() < 0.35:
                    self.compromised_for[i] = max(0, self.compromised_for[i] - 2)
            elif action_name == "lowpower":
                if self.config.enable_recharge and self._in_sunlight():
                    self.energy[i] = min(self.config.energy_budget, self.energy[i] + self.config.recharge_rate)

        if self.config.enable_recharge and self._in_sunlight():
            for i in range(n):
                if self._is_alive(i):
                    self.energy[i] = min(self.config.energy_budget, self.energy[i] + 0.15 * self.config.recharge_rate)

        self._update_tasks()
        self.update_comm_graph()
        self.compromised_for = np.maximum(0, self.compromised_for - 1)
        self.scan_boost = np.maximum(0, self.scan_boost - 1)
        self.t += 1

        alive = (self.energy > 0).sum()
        isolated = self._isolated_count()
        failures = float(n - alive)

        components = {
            "task": float(serviced.sum()),
            "delivery": float(delivered.sum()),
            "energy": float(energy_spent.sum()),
            "isolation": float(isolated),
            "failure": failures,
            "cyber": float(cyber_penalty.sum()),
        }
        self.last_reward_components = components
        shared = compute_shared_reward(components, self.config.reward_weights)
        self.last_reward = shared

        rewards = {}
        for i, name in enumerate(agent_names):
            if self.config.reward_mode == "local":
                local_components = {
                    "task": float(serviced[i]),
                    "delivery": float(delivered[i]),
                    "energy": float(energy_spent[i]),
                    "isolation": float(1.0 if self._is_alive(i) and self.comm_adj[i].sum() == 0 else 0.0),
                    "failure": float(0.0 if self._is_alive(i) else 1.0),
                    "cyber": float(cyber_penalty[i]),
                }
                rewards[name] = compute_shared_reward(local_components, self.config.reward_weights)
            else:
                rewards[name] = shared

        terminated = self._mission_failed()
        trunc = self.t >= self.config.max_steps
        terminations = {name: terminated for name in agent_names}
        truncations = {name: trunc for name in agent_names}
        infos = {name: self._build_info(i) for i, name in enumerate(agent_names)}
        if terminated or trunc:
            episode = {
                "steps": self.t,
                "delivered": self.delivered_total,
                "last_reward": shared,
                "alive": int(alive),
            }
            for info in infos.values():
                info["episode"] = episode
        return rewards, terminations, truncations, infos

    def _update_tasks(self) -> None:
        for idx, task in enumerate(self.tasks):
            if task.active:
                task.age += 1
                if task.age > 25:
                    task.active = False
                elif self.config.task_priority_mode == "dynamic":
                    task.priority = float(np.clip(task.priority + self.rng.uniform(-0.05, 0.08), 0.1, 1.0))
            elif self.rng.random() < self.config.task_spawn_rate:
                self.tasks[idx] = self._spawn_task()

    def _has_path_to_ground(self, src: int) -> bool:
        for j in range(self.num_agents):
            if self.comm_adj[src, j] and np.abs(self.positions[j] - self.ground).sum() <= 1:
                return True
        return False

    def _isolated_count(self) -> int:
        cnt = 0
        for i in range(self.num_agents):
            if self._is_alive(i) and self.comm_adj[i].sum() == 0:
                cnt += 1
        return cnt

    def _mission_failed(self) -> bool:
        alive = int((self.energy > 0).sum())
        if alive == 0:
            return True
        if alive <= max(1, self.num_agents // 4):
            return True
        return False

    def observe(self, i: int) -> np.ndarray:
        gs = float(self.config.grid_size - 1)
        energy_norm = self.energy[i] / max(self.config.energy_budget, 1e-6)
        pos = self.positions[i] / max(gs, 1.0)
        cyber = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if self.compromised_for[i] > 0:
            cyber = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif self.scan_boost[i] > 0:
            cyber = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        deg_norm = float(self.comm_adj[i].sum()) / max(1, self.num_agents - 1)
        local_count = 0.0
        local_prio = 0.0
        for t in self.tasks:
            if t.active and np.abs(self.positions[i] - np.array([t.x, t.y])).sum() <= 2:
                local_count += 1.0
                local_prio += t.priority
        local_count = min(local_count / max(1.0, self.config.num_tasks), 1.0)
        local_prio = min(local_prio / max(1.0, self.config.num_tasks), 1.0)
        buffer_norm = min(self.buffered_data[i] / 5.0, 1.0)

        ang = 2.0 * np.pi * ((self.t % self.config.sunlight_period) / self.config.sunlight_period)
        t_sin = np.sin(ang)
        t_cos = np.cos(ang)
        compromised_neighbors = 0.0
        neighbors = np.where(self.comm_adj[i])[0]
        if len(neighbors) > 0:
            compromised_neighbors = float((self.compromised_for[neighbors] > 0).sum()) / len(neighbors)
        alive_frac = float((self.energy > 0).sum()) / self.num_agents

        obs = np.array([
            energy_norm,
            pos[0],
            pos[1],
            cyber[0],
            cyber[1],
            cyber[2],
            deg_norm,
            local_count,
            local_prio,
            buffer_norm,
            t_sin,
            t_cos,
            compromised_neighbors,
            alive_frac,
        ], dtype=np.float32)

        if self.compromised_for[i] > 0 and self.config.spoof_mode == "obs_spoof":
            obs = obs.copy()
            obs[0] = float(self.rng.uniform(0.0, 1.0))
            obs[7:9] = self.rng.uniform(0.0, 1.0, size=(2,))
        return obs

    def _build_info(self, i: int) -> dict[str, Any]:
        return {
            "energy": float(self.energy[i]),
            "compromised": bool(self.compromised_for[i] > 0),
            "local_degree": int(self.comm_adj[i].sum()),
            "buffered_data": float(self.buffered_data[i]),
            "time": self.t,
            "reward_components": dict(self.last_reward_components),
        }
