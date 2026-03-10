from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.reward import compute_shared_reward
from orbital.envs.core.spaces import ACTION_MAP


@dataclass
class Task:
    theta: float
    radius: float
    phi: float
    priority: float
    active: bool = True
    age: int = 0


@dataclass
class DebrisCloud:
    theta: float
    radius: float
    phi: float
    spread: float
    density: float
    drift: float


class OrbitalCore:
    """Core ORBITAL transition model shared by AEC and Parallel wrappers.

    High-level step semantics:
    1) apply agent actions under energy/cyber constraints,
    2) update orbital motion and communication graph,
    3) update tasks and mission-level reward components.
    """

    def __init__(self, config: OrbitalConfig):
        self.config = config
        self.num_agents = config.num_satellites
        # compatibility with external callers
        self.ground = np.array([0, 0], dtype=np.int32)
        self.ground_thetas = np.array(config.ground_station_thetas, dtype=np.float32)
        self.ground_phis = np.array(config.ground_station_phis, dtype=np.float32)
        self.ground_theta = float(self.ground_thetas[0])
        self.ground_vectors = np.zeros((len(self.ground_thetas), 3), dtype=np.float32)
        if self.config.world_dim == 3:
            for idx, (th, ph) in enumerate(zip(self.ground_thetas, self.ground_phis)):
                cphi = np.cos(float(ph))
                self.ground_vectors[idx] = np.array(
                    [
                        self.config.earth_radius * np.cos(float(th)) * cphi,
                        self.config.earth_radius * np.sin(float(th)) * cphi,
                        self.config.earth_radius * np.sin(float(ph)),
                    ],
                    dtype=np.float32,
                )
        self.rng = np.random.default_rng()
        self.reset(seed=None)

    def reset(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.orbit_theta = self.rng.uniform(
            0.0, 2.0 * np.pi, size=(self.num_agents,)).astype(np.float32)
        if self.config.world_dim == 3:
            self.orbit_phi = self.rng.uniform(
                -self.config.inclination_max,
                self.config.inclination_max,
                size=(self.num_agents,),
            ).astype(np.float32)
        else:
            self.orbit_phi = np.zeros((self.num_agents,), dtype=np.float32)
        self.orbit_radius = self.rng.uniform(
            self.config.orbit_min_radius, self.config.orbit_max_radius, size=(
                self.num_agents,)
        ).astype(np.float32)
        self.positions = np.zeros((self.num_agents, self.config.world_dim), dtype=np.float32)
        self._refresh_cartesian_positions()
        self.energy = np.full((self.num_agents,),
                              self.config.energy_budget, dtype=np.float32)
        self.compromised_for = np.zeros((self.num_agents,), dtype=np.int32)
        self.scan_boost = np.zeros((self.num_agents,), dtype=np.int32)
        self.buffered_data = np.zeros((self.num_agents,), dtype=np.float32)
        self.tasks = [self._spawn_task() for _ in range(self.config.num_tasks)]
        self.debris_clouds = [self._spawn_debris_cloud() for _ in range(self.config.num_debris_clouds)]
        self.comm_adj = np.zeros(
            (self.num_agents, self.num_agents), dtype=np.bool_)
        self.last_reward_components = {"task": 0.0, "delivery": 0.0,
                                       "energy": 0.0, "isolation": 0.0, "failure": 0.0, "cyber": 0.0,
                                       "debris_risk": 0.0, "collision": 0.0}
        self.last_reward = 0.0
        self.delivered_total = 0.0
        self.update_comm_graph()

    def _spawn_task(self) -> Task:
        prio = float(self.rng.uniform(0.2, 1.0))
        return Task(
            theta=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            radius=float(self.rng.uniform(
                self.config.orbit_min_radius, self.config.orbit_max_radius)),
            phi=float(self.rng.uniform(-self.config.inclination_max, self.config.inclination_max))
            if self.config.world_dim == 3
            else 0.0,
            priority=prio,
            active=True,
            age=0,
        )

    def _spawn_debris_cloud(self) -> DebrisCloud:
        return DebrisCloud(
            theta=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            radius=float(self.rng.uniform(self.config.orbit_min_radius, self.config.orbit_max_radius)),
            phi=float(self.rng.uniform(-self.config.inclination_max, self.config.inclination_max))
            if self.config.world_dim == 3
            else 0.0,
            spread=float(self.rng.uniform(self.config.debris_spread_min, self.config.debris_spread_max)),
            density=float(self.rng.uniform(0.35, 1.0)),
            drift=float(self.rng.normal(0.0, self.config.debris_drift_std)),
        )

    def _is_alive(self, i: int) -> bool:
        return self.energy[i] > 0.0

    def update_comm_graph(self) -> None:
        n = self.num_agents
        adj = np.zeros((n, n), dtype=np.bool_)
        comm_dist = self._comm_distance_threshold()
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(
                    self.positions[i] - self.positions[j]))
                los_clear = not self._segment_intersects_earth(self.positions[i], self.positions[j])
                if dist <= comm_dist and los_clear and self.rng.random() > self.config.p_link_drop:
                    adj[i, j] = True
                    adj[j, i] = True
        self.comm_adj = adj

    def _in_sunlight(self) -> bool:
        phase = self.t % self.config.sunlight_period
        return phase < (self.config.sunlight_period // 2)

    def _cartesian_from_orbit(self, theta: float, radius: float, phi: float = 0.0) -> np.ndarray:
        if self.config.world_dim == 3:
            cphi = np.cos(phi)
            return np.array(
                [radius * np.cos(theta) * cphi, radius * np.sin(theta) * cphi, radius * np.sin(phi)],
                dtype=np.float32,
            )
        return np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)

    def _refresh_cartesian_positions(self) -> None:
        for i in range(self.num_agents):
            self.positions[i] = self._cartesian_from_orbit(
                float(self.orbit_theta[i]),
                float(self.orbit_radius[i]),
                float(self.orbit_phi[i]),
            )

    def _wrap_angle(self, theta: float) -> float:
        return float(theta % (2.0 * np.pi))

    def _angle_delta(self, a0: float, a1: float) -> float:
        return float(((a1 - a0 + np.pi) % (2.0 * np.pi)) - np.pi)

    def _comm_distance_threshold(self) -> float:
        radial_span = self.config.orbit_max_radius - self.config.orbit_min_radius
        band_step = radial_span / max(1.0, float(self.config.grid_size - 1))
        return max(0.35, float(self.config.comm_radius) * band_step)

    def _sensing_distance_threshold(self) -> float:
        radial_span = self.config.orbit_max_radius - self.config.orbit_min_radius
        band_step = radial_span / max(1.0, float(self.config.grid_size - 1))
        return max(0.25, 1.4 * band_step)

    def _direct_ground_contact(self, i: int) -> bool:
        if not self._is_alive(i):
            return False
        if self.config.world_dim == 3:
            sat_vec = self.positions[i].astype(np.float64)
            sat_norm = float(np.linalg.norm(sat_vec))
            if sat_norm < 1e-9:
                return False
            sat_unit = sat_vec / sat_norm
            cos_th = np.cos(self.config.ground_contact_angle)
            for g in self.ground_vectors:
                g_vec = g.astype(np.float64)
                g_norm = float(np.linalg.norm(g_vec))
                if g_norm < 1e-9:
                    continue
                g_unit = g_vec / g_norm
                if float(np.dot(sat_unit, g_unit)) >= cos_th:
                    return True
            return False
        sat_theta = float(self.orbit_theta[i])
        for gs_theta in self.ground_thetas:
            dtheta = abs(self._angle_delta(sat_theta, float(gs_theta)))
            if dtheta <= self.config.ground_contact_angle:
                return True
        return False

    def _segment_intersects_earth(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        p1f = p1.astype(np.float64)
        p2f = p2.astype(np.float64)
        d = p2f - p1f
        a = float(np.dot(d, d))
        if a < 1e-12:
            return False
        b = 2.0 * float(np.dot(p1f, d))
        c = float(np.dot(p1f, p1f) - (self.config.earth_radius ** 2))
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = float(np.sqrt(max(0.0, disc)))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        if t1 > t2:
            t1, t2 = t2, t1
        eps = 1e-6
        return (t1 < 1.0 - eps) and (t2 > eps)

    def _action_name(self, a: int) -> str:
        return ACTION_MAP.get(int(a), "idle")

    def _apply_orbit_shift(self, i: int, delta_radius: float) -> None:
        nr = float(self.orbit_radius[i] + delta_radius)
        nr = float(np.clip(nr, self.config.orbit_min_radius,
                   self.config.orbit_max_radius))
        self.orbit_radius[i] = nr

    def _propagate_kepler(self, i: int) -> None:
        # Kepler-like angular velocity: omega ~ r^(-3/2)
        r = max(1e-6, float(self.orbit_radius[i]))
        omega = self.config.kepler_constant / (r ** 1.5)
        self.orbit_theta[i] = self._wrap_angle(
            float(self.orbit_theta[i] + omega))

    def _update_debris_clouds(self) -> None:
        if not self.config.enable_debris:
            return
        for idx, cloud in enumerate(self.debris_clouds):
            if cloud.density <= 1e-4:
                if self.rng.random() < self.config.debris_spawn_rate:
                    self.debris_clouds[idx] = self._spawn_debris_cloud()
                continue
            cloud.theta = self._wrap_angle(cloud.theta + cloud.drift + float(self.rng.normal(0.0, 0.01)))
            if self.config.world_dim == 3:
                cloud.phi = float(
                    np.clip(
                        cloud.phi + float(self.rng.normal(0.0, self.config.debris_drift_std * 0.5)),
                        -self.config.inclination_max,
                        self.config.inclination_max,
                    )
                )
            cloud.radius = float(np.clip(
                cloud.radius + float(self.rng.normal(0.0, self.config.debris_drift_std)),
                self.config.orbit_min_radius,
                self.config.orbit_max_radius,
            ))
            cloud.spread = float(np.clip(
                cloud.spread + float(self.rng.normal(0.0, 0.02)),
                self.config.debris_spread_min,
                self.config.debris_spread_max,
            ))
            cloud.density = max(0.0, cloud.density * (1.0 - self.config.debris_decay))
            if self.rng.random() < self.config.debris_spawn_rate * 0.25:
                cloud.density = min(1.0, cloud.density + float(self.rng.uniform(0.05, 0.2)))

    def _local_debris_density(self, i: int) -> float:
        if not self.config.enable_debris or len(self.debris_clouds) == 0 or not self._is_alive(i):
            return 0.0
        sat_theta = float(self.orbit_theta[i])
        sat_radius = float(self.orbit_radius[i])
        sat_phi = float(self.orbit_phi[i])
        density = 0.0
        for cloud in self.debris_clouds:
            if cloud.density <= 1e-4:
                continue
            d_theta = self._angle_delta(sat_theta, cloud.theta)
            d_radius = sat_radius - cloud.radius
            d_phi = sat_phi - cloud.phi if self.config.world_dim == 3 else 0.0
            dist = float(np.sqrt(d_theta * d_theta + d_radius * d_radius + d_phi * d_phi))
            scale = max(1e-3, cloud.spread)
            density += cloud.density * float(np.exp(-0.5 * (dist / scale) ** 2))
        return float(np.clip(density, 0.0, 1.5))

    def _local_pc_estimate(self, i: int) -> float:
        base = self._local_debris_density(i)
        if self.compromised_for[i] > 0:
            base *= 1.10
        return float(np.clip(self.config.debris_risk_gain * base, 0.0, 1.0))

    def step(self, actions: dict[str, int], agent_names: list[str]) -> tuple[dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        """Advance one synchronized fleet step.

        `Observe` can increase buffered data, but only `Relay` can convert it
        into delivered mission value when communication-to-ground is available.
        This decoupling is the key long-horizon coordination tension in ORBITAL.
        """

        n = self.num_agents
        serviced = np.zeros((n,), dtype=np.float32)
        delivered = np.zeros((n,), dtype=np.float32)
        energy_spent = np.zeros((n,), dtype=np.float32)
        cyber_penalty = np.zeros((n,), dtype=np.float32)
        debris_risk = np.zeros((n,), dtype=np.float32)
        collision = np.zeros((n,), dtype=np.float32)
        executed_actions = ["idle"] * n

        # adversarial event
        if self.rng.random() < self.config.adversarial_rate:
            healthy = np.where(self.compromised_for <= 0)[0]
            if len(healthy) > 0:
                idx = int(self.rng.choice(healthy))
                self.compromised_for[idx] = self.config.compromise_duration

        for i, name in enumerate(agent_names):
            act = int(actions.get(name, 6))
            if not self._is_alive(i):
                continue

            if self.compromised_for[i] > 0 and self.config.spoof_mode == "action_noise" and self.rng.random() < 0.20:
                act = int(self.rng.integers(0, len(ACTION_MAP)))
                cyber_penalty[i] += 0.5

            action_name = self._action_name(act)
            executed_actions[i] = action_name
            cost = self.config.energy_costs.get(
                action_name, self.config.energy_costs["idle"])
            self.energy[i] -= cost
            energy_spent[i] += cost

            if self.energy[i] <= 0:
                self.energy[i] = 0.0
                continue

            if action_name == "orbit_down":
                self._apply_orbit_shift(i, -self.config.orbit_shift_step)
            elif action_name == "orbit_up":
                self._apply_orbit_shift(i, self.config.orbit_shift_step)
            elif action_name == "observe":
                sat_pos = self.positions[i]
                sensing_dist = self._sensing_distance_threshold()
                for task in self.tasks:
                    if not task.active:
                        continue
                    task_pos = self._cartesian_from_orbit(
                        task.theta, task.radius, task.phi)
                    d = float(np.linalg.norm(sat_pos - task_pos))
                    if d <= sensing_dist:
                        task.active = False
                        serviced[i] += task.priority
                        self.buffered_data[i] += task.priority
                        break
            elif action_name == "relay":
                if self.buffered_data[i] > 0.0:
                    at_ground = self._direct_ground_contact(i)
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
                    self.compromised_for[i] = max(
                        0, self.compromised_for[i] - 2)
            elif action_name == "lowpower":
                if self.config.enable_recharge and self._in_sunlight():
                    self.energy[i] = min(
                        self.config.energy_budget, self.energy[i] + self.config.recharge_rate)

            self._propagate_kepler(i)

        if self.config.enable_recharge and self._in_sunlight():
            for i in range(n):
                if self._is_alive(i):
                    self.energy[i] = min(
                        self.config.energy_budget, self.energy[i] + 0.15 * self.config.recharge_rate)

        self._refresh_cartesian_positions()
        self._update_debris_clouds()
        if self.config.enable_debris:
            for i in range(n):
                if not self._is_alive(i):
                    continue
                pc = self._local_pc_estimate(i)
                if executed_actions[i] in {"orbit_down", "orbit_up"}:
                    pc *= (1.0 - self.config.debris_mitigation_factor)
                debris_risk[i] = pc
                if pc >= self.config.pc_alert_threshold and executed_actions[i] not in {"orbit_down", "orbit_up"}:
                    p_collision = self.config.pc_collision_scale * pc
                    if self.rng.random() < p_collision:
                        self.energy[i] = 0.0
                        self.buffered_data[i] = 0.0
                        collision[i] = 1.0
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
            "debris_risk": float(debris_risk.sum()),
            "collision": float(collision.sum()),
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
                    "debris_risk": float(debris_risk[i]),
                    "collision": float(collision[i]),
                }
                rewards[name] = compute_shared_reward(
                    local_components, self.config.reward_weights)
            else:
                rewards[name] = shared

        terminated = self._mission_failed()
        trunc = self.t >= self.config.max_steps
        terminations = {name: terminated for name in agent_names}
        truncations = {name: trunc for name in agent_names}
        infos = {name: self._build_info(i)
                 for i, name in enumerate(agent_names)}
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
                    task.priority = float(
                        np.clip(task.priority + self.rng.uniform(-0.05, 0.08), 0.1, 1.0))
            elif self.rng.random() < self.config.task_spawn_rate:
                self.tasks[idx] = self._spawn_task()

    def _has_path_to_ground(self, src: int) -> bool:
        for j in range(self.num_agents):
            if self.comm_adj[src, j] and self._direct_ground_contact(j):
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
        energy_norm = self.energy[i] / max(self.config.energy_budget, 1e-6)
        theta_norm = float(self.orbit_theta[i]) / (2.0 * np.pi)
        phi_norm = (
            (float(self.orbit_phi[i]) + self.config.inclination_max)
            / max(1e-6, 2.0 * self.config.inclination_max)
            if self.config.world_dim == 3
            else 0.0
        )
        radius_norm = (float(self.orbit_radius[i]) - self.config.orbit_min_radius) / max(
            1e-6, self.config.orbit_max_radius - self.config.orbit_min_radius
        )
        radius_norm = float(np.clip(radius_norm, 0.0, 1.0))
        cyber = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if self.compromised_for[i] > 0:
            cyber = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif self.scan_boost[i] > 0:
            cyber = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        deg_norm = float(self.comm_adj[i].sum()) / max(1, self.num_agents - 1)
        local_count = 0.0
        local_prio = 0.0
        sensing_dist = self._sensing_distance_threshold()
        sat_pos = self.positions[i]
        for t in self.tasks:
            if not t.active:
                continue
            task_pos = self._cartesian_from_orbit(t.theta, t.radius, t.phi)
            if float(np.linalg.norm(sat_pos - task_pos)) <= (1.75 * sensing_dist):
                local_count += 1.0
                local_prio += t.priority
        local_count = min(local_count / max(1.0, self.config.num_tasks), 1.0)
        local_prio = min(local_prio / max(1.0, self.config.num_tasks), 1.0)
        buffer_norm = min(self.buffered_data[i] / 5.0, 1.0)
        local_debris_density = min(self._local_debris_density(i), 1.0)
        local_pc = self._local_pc_estimate(i)

        ang = 2.0 * np.pi * \
            ((self.t % self.config.sunlight_period) / self.config.sunlight_period)
        t_sin = np.sin(ang)
        t_cos = np.cos(ang)
        compromised_neighbors = 0.0
        neighbors = np.where(self.comm_adj[i])[0]
        if len(neighbors) > 0:
            compromised_neighbors = float(
                (self.compromised_for[neighbors] > 0).sum()) / len(neighbors)
        alive_frac = float((self.energy > 0).sum()) / self.num_agents

        obs = np.array([
            energy_norm,
            theta_norm,
            radius_norm,
            cyber[0],
            cyber[1],
            cyber[2],
            deg_norm,
            local_count,
            local_prio,
            buffer_norm,
            local_debris_density,
            local_pc,
            t_sin,
            phi_norm if self.config.world_dim == 3 else t_cos,
            compromised_neighbors,
            alive_frac,
        ], dtype=np.float32)

        if self.compromised_for[i] > 0 and self.config.spoof_mode == "obs_spoof":
            obs = obs.copy()
            obs[0] = float(self.rng.uniform(0.0, 1.0))
            obs[1] = float(self.rng.uniform(0.0, 1.0))
            obs[6:9] = self.rng.uniform(0.0, 1.0, size=(3,))
            obs[10:12] = self.rng.uniform(0.0, 1.0, size=(2,))
        return obs

    def _build_info(self, i: int) -> dict[str, Any]:
        return {
            "energy": float(self.energy[i]),
            "compromised": bool(self.compromised_for[i] > 0),
            "local_degree": int(self.comm_adj[i].sum()),
            "buffered_data": float(self.buffered_data[i]),
            "theta": float(self.orbit_theta[i]),
            "phi": float(self.orbit_phi[i]),
            "radius": float(self.orbit_radius[i]),
            "local_debris_density": float(self._local_debris_density(i)),
            "local_pc_estimate": float(self._local_pc_estimate(i)),
            "time": self.t,
            "reward_components": dict(self.last_reward_components),
        }
