from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.reward import compute_shared_reward
from orbital.envs.core.spaces import ACTION_MAP


@dataclass
class KeplerOrbit:
    semi_major_axis: float
    eccentricity: float
    mean_anomaly: float
    arg_periapsis: float
    inclination: float
    raan: float


@dataclass
class Task:
    orbit: KeplerOrbit
    priority: float
    theta: float = 0.0
    radius: float = 0.0
    phi: float = 0.0
    active: bool = True
    age: int = 0


@dataclass
class DebrisCloud:
    orbit: KeplerOrbit
    spread: float
    density: float
    theta: float = 0.0
    radius: float = 0.0
    phi: float = 0.0


class OrbitalCore:
    """ORBITAL V2 transition model.

    V2 makes task knowledge distributed, separates ground and satellite relay,
    and models energy, health, debris, orbit, solar recharge, and malware as
    distinct mission constraints.
    """

    def __init__(self, config: OrbitalConfig):
        self.config = config
        self.num_agents = config.num_satellites
        self.ground = np.array([0, 0], dtype=np.int32)
        self.ground_thetas = np.array(
            config.ground_station_thetas, dtype=np.float32)
        self.ground_phis = np.array(
            config.ground_station_phis, dtype=np.float32)
        self.ground_theta = float(self.ground_thetas[0])
        self.ground_vectors = np.zeros(
            (len(self.ground_thetas), 3), dtype=np.float32)
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
        sampled_orbits = [self._sample_orbit() for _ in range(self.num_agents)]
        self.orbit_semi_major_axis = np.array(
            [orbit.semi_major_axis for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_eccentricity = np.array(
            [orbit.eccentricity for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_mean_anomaly = np.array(
            [orbit.mean_anomaly for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_arg_periapsis = np.array(
            [orbit.arg_periapsis for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_inclination = np.array(
            [orbit.inclination for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_raan = np.array(
            [orbit.raan for orbit in sampled_orbits], dtype=np.float32)
        self.orbit_theta = np.zeros((self.num_agents,), dtype=np.float32)
        self.orbit_phi = np.zeros((self.num_agents,), dtype=np.float32)
        self.orbit_radius = np.zeros((self.num_agents,), dtype=np.float32)
        self.positions = np.zeros(
            (self.num_agents, self.config.world_dim), dtype=np.float32)
        self._refresh_satellite_positions()

        self.energy = np.full((self.num_agents,),
                              self.config.energy_budget, dtype=np.float32)
        self.health = np.full((self.num_agents,),
                              self.config.health_budget, dtype=np.float32)
        self.compromised_for = np.zeros((self.num_agents,), dtype=np.int32)
        self.malware_awake = np.zeros((self.num_agents,), dtype=np.bool_)
        self.jammed = np.zeros((self.num_agents,), dtype=np.bool_)
        self.last_action_forced = np.zeros((self.num_agents,), dtype=np.bool_)
        self.scan_boost = np.zeros((self.num_agents,), dtype=np.int32)
        self.buffered_data = np.zeros((self.num_agents,), dtype=np.float32)

        self.tasks = [self._spawn_task() for _ in range(self.config.num_tasks)]
        self.known_tasks = np.zeros(
            (self.num_agents, self.config.num_tasks), dtype=np.bool_)
        self.station_known_tasks = np.array(
            [self.config.task_knowledge_mode ==
                "ground_catalog" and task.active for task in self.tasks],
            dtype=np.bool_,
        )
        self.debris_clouds = [self._spawn_debris_cloud()
                              for _ in range(self.config.num_debris_clouds)]
        self.comm_adj = np.zeros(
            (self.num_agents, self.num_agents), dtype=np.bool_)
        self.last_executed_actions = ["idle"] * self.num_agents
        self.last_reward_components = self._empty_components()
        self.last_reward = 0.0
        self.delivered_total = 0.0
        self.observed_total = 0.0
        self.knowledge_shared_total = 0.0
        self.jam_count = 0
        self.forced_action_count = 0
        self.update_comm_graph()
        self._refresh_task_knowledge()

    def _empty_components(self) -> dict[str, float]:
        return {
            "task": 0.0,
            "delivery": 0.0,
            "ground_task_intake": 0.0,
            "knowledge": 0.0,
            "energy": 0.0,
            "overflow": 0.0,
            "data_loss": 0.0,
            "health": 0.0,
            "isolation": 0.0,
            "failure": 0.0,
            "cyber": 0.0,
            "jam": 0.0,
            "forced_action": 0.0,
            "atmospheric_drag": 0.0,
            "debris_risk": 0.0,
            "collision": 0.0,
        }

    def _spawn_task(self) -> Task:
        task = Task(
            orbit=self._sample_orbit(),
            priority=float(self.rng.uniform(0.2, 1.0)),
            active=True,
            age=0,
        )
        self._refresh_body_coordinates(task)
        return task

    def _spawn_debris_cloud(self) -> DebrisCloud:
        cloud = DebrisCloud(
            orbit=self._sample_orbit(),
            spread=float(self.rng.uniform(
                self.config.debris_spread_min, self.config.debris_spread_max)),
            density=float(self.rng.uniform(0.35, 1.0)),
        )
        self._refresh_body_coordinates(cloud)
        return cloud

    def _is_alive(self, i: int) -> bool:
        return self.health[i] > 0.0

    def _is_powered(self, i: int) -> bool:
        return self._is_alive(i) and self.energy[i] > 0.0

    def update_comm_graph(self) -> None:
        n = self.num_agents
        adj = np.zeros((n, n), dtype=np.bool_)
        comm_dist = self._comm_distance_threshold()
        for i in range(n):
            if not self._is_alive(i):
                continue
            for j in range(i + 1, n):
                if not self._is_alive(j):
                    continue
                dist = float(np.linalg.norm(
                    self.positions[i] - self.positions[j]))
                los_clear = not self._segment_intersects_earth(
                    self.positions[i], self.positions[j])
                if dist <= comm_dist and los_clear and self.rng.random() > self.config.p_link_drop:
                    adj[i, j] = True
                    adj[j, i] = True
        self.comm_adj = adj

    def _in_sunlight(self, i: int | None = None) -> bool:
        if i is None:
            return True
        return bool(self.positions[i][1] < 0.0)

    def _cartesian_from_orbit(self, theta: float, radius: float, phi: float = 0.0) -> np.ndarray:
        if self.config.world_dim == 3:
            cphi = np.cos(phi)
            return np.array(
                [radius * np.cos(theta) * cphi, radius *
                 np.sin(theta) * cphi, radius * np.sin(phi)],
                dtype=np.float32,
            )
        return np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)

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

    def _discovery_distance_threshold(self) -> float:
        return 1.75 * self._sensing_distance_threshold()

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
                if float(np.dot(sat_unit, g_vec / g_norm)) >= cos_th:
                    return True
            return False
        sat_theta = float(self.orbit_theta[i])
        return any(abs(self._angle_delta(sat_theta, float(gs_theta))) <= self.config.ground_contact_angle for gs_theta in self.ground_thetas)

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

    def _sample_orbit(self) -> KeplerOrbit:
        eccentricity = float(self.rng.uniform(
            self.config.eccentricity_min, self.config.eccentricity_max))
        lo, hi = self._semi_major_axis_bounds(eccentricity)
        return KeplerOrbit(
            semi_major_axis=float(self.rng.uniform(lo, hi)),
            eccentricity=eccentricity,
            mean_anomaly=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            arg_periapsis=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            inclination=float(self.rng.uniform(
                -self.config.inclination_max, self.config.inclination_max))
            if self.config.world_dim == 3
            else 0.0,
            raan=float(self.rng.uniform(0.0, 2.0 * np.pi))
            if self.config.world_dim == 3
            else 0.0,
        )

    def _semi_major_axis_bounds(self, eccentricity: float) -> tuple[float, float]:
        lo = self.config.orbit_min_radius / max(1e-6, 1.0 - eccentricity)
        hi = self.config.orbit_max_radius / max(1e-6, 1.0 + eccentricity)
        return float(lo), float(hi)

    def _apply_orbit_shift(self, i: int, delta_axis: float) -> None:
        lo, hi = self._semi_major_axis_bounds(float(self.orbit_eccentricity[i]))
        self.orbit_semi_major_axis[i] = float(np.clip(
            float(self.orbit_semi_major_axis[i]) + delta_axis, lo, hi))

    def _kepler_mean_motion(self, semi_major_axis: float) -> float:
        a = max(1e-6, float(semi_major_axis))
        return float(self.config.kepler_constant / (a ** 1.5))

    def _solve_eccentric_anomaly(self, mean_anomaly: float, eccentricity: float) -> float:
        mean = self._wrap_angle(mean_anomaly)
        eccentric = mean if eccentricity < 0.8 else np.pi
        for _ in range(8):
            residual = eccentric - eccentricity * np.sin(eccentric) - mean
            derivative = 1.0 - eccentricity * np.cos(eccentric)
            eccentric -= residual / max(1e-8, derivative)
        return float(eccentric)

    def _coordinates_from_elements(self, orbit: KeplerOrbit) -> tuple[np.ndarray, float, float, float]:
        eccentric_anomaly = self._solve_eccentric_anomaly(
            orbit.mean_anomaly, orbit.eccentricity)
        x_perifocal = orbit.semi_major_axis * \
            (np.cos(eccentric_anomaly) - orbit.eccentricity)
        y_perifocal = orbit.semi_major_axis * \
            np.sqrt(1.0 - orbit.eccentricity ** 2) * np.sin(eccentric_anomaly)

        cos_arg = np.cos(orbit.arg_periapsis)
        sin_arg = np.sin(orbit.arg_periapsis)
        x_node = cos_arg * x_perifocal - sin_arg * y_perifocal
        y_node = sin_arg * x_perifocal + cos_arg * y_perifocal

        if self.config.world_dim == 3:
            cos_inc = np.cos(orbit.inclination)
            sin_inc = np.sin(orbit.inclination)
            cos_raan = np.cos(orbit.raan)
            sin_raan = np.sin(orbit.raan)
            x_plane = x_node
            y_plane = y_node * cos_inc
            z = y_node * sin_inc
            x = cos_raan * x_plane - sin_raan * y_plane
            y = sin_raan * x_plane + cos_raan * y_plane
            vec = np.array([x, y, z], dtype=np.float32)
        else:
            vec = np.array([x_node, y_node], dtype=np.float32)

        radius = float(np.linalg.norm(vec))
        theta = self._wrap_angle(float(np.arctan2(vec[1], vec[0])))
        phi = float(np.arcsin(np.clip(float(vec[2]) / max(1e-6, radius), -1.0, 1.0))) \
            if self.config.world_dim == 3 else 0.0
        return vec, theta, radius, phi

    def _satellite_orbit(self, i: int) -> KeplerOrbit:
        return KeplerOrbit(
            semi_major_axis=float(self.orbit_semi_major_axis[i]),
            eccentricity=float(self.orbit_eccentricity[i]),
            mean_anomaly=float(self.orbit_mean_anomaly[i]),
            arg_periapsis=float(self.orbit_arg_periapsis[i]),
            inclination=float(self.orbit_inclination[i]),
            raan=float(self.orbit_raan[i]),
        )

    def _refresh_satellite_positions(self) -> None:
        for i in range(self.num_agents):
            vec, theta, radius, phi = self._coordinates_from_elements(
                self._satellite_orbit(i))
            self.positions[i] = vec
            self.orbit_theta[i] = theta
            self.orbit_radius[i] = radius
            self.orbit_phi[i] = phi

    def _refresh_cartesian_positions(self) -> None:
        self._refresh_satellite_positions()

    def _refresh_body_coordinates(self, body: Task | DebrisCloud) -> None:
        _, body.theta, body.radius, body.phi = self._coordinates_from_elements(
            body.orbit)

    def _propagate_body(self, body: Task | DebrisCloud) -> None:
        body.orbit.mean_anomaly = self._wrap_angle(
            body.orbit.mean_anomaly + self._kepler_mean_motion(body.orbit.semi_major_axis))
        self._refresh_body_coordinates(body)

    def _propagate_kepler(self, i: int) -> None:
        mean_motion = self._kepler_mean_motion(float(self.orbit_semi_major_axis[i]))
        self.orbit_mean_anomaly[i] = self._wrap_angle(
            float(self.orbit_mean_anomaly[i] + mean_motion))

    def _high_orbit_factor(self, i: int) -> float:
        high_start = self.config.orbit_max_radius - self.config.high_orbit_margin
        if self.orbit_radius[i] <= high_start:
            return 0.0
        return float(np.clip((self.orbit_radius[i] - high_start) / max(1e-6, self.config.high_orbit_margin), 0.0, 1.0))

    def _is_low_orbit(self, i: int) -> bool:
        return bool(self.orbit_radius[i] <= self.config.orbit_min_radius + self.config.low_orbit_margin)

    def _update_debris_clouds(self) -> None:
        if not self.config.enable_debris:
            return
        for idx, cloud in enumerate(self.debris_clouds):
            if cloud.density <= 1e-4:
                if self.rng.random() < self.config.debris_spawn_rate:
                    self.debris_clouds[idx] = self._spawn_debris_cloud()
                continue
            self._propagate_body(cloud)
            cloud.spread = float(np.clip(
                cloud.spread + float(self.rng.normal(0.0, 0.02)),
                self.config.debris_spread_min,
                self.config.debris_spread_max,
            ))
            cloud.density = max(0.0, cloud.density *
                                (1.0 - self.config.debris_decay))
            if self.rng.random() < self.config.debris_spawn_rate * 0.25:
                cloud.density = min(1.0, cloud.density +
                                    float(self.rng.uniform(0.05, 0.2)))

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
            dist = float(np.sqrt(d_theta * d_theta +
                         d_radius * d_radius + d_phi * d_phi))
            density += cloud.density * \
                float(np.exp(-0.5 * (dist / max(1e-3, cloud.spread)) ** 2))
        return float(np.clip(density, 0.0, 1.5))

    def _local_pc_estimate(self, i: int) -> float:
        base = self._local_debris_density(i)
        if self.compromised_for[i] > 0:
            base *= 1.10
        return float(np.clip(self.config.debris_risk_gain * base, 0.0, 1.0))

    def _refresh_task_knowledge(self) -> float:
        gained = 0.0
        if self.config.task_knowledge_mode == "ground_catalog":
            for task_idx, task in enumerate(self.tasks):
                self.station_known_tasks[task_idx] = bool(task.active)
        if not self.config.enable_local_task_discovery:
            return gained
        discover_dist = self._discovery_distance_threshold()
        for task_idx, task in enumerate(self.tasks):
            if not task.active:
                continue
            task_pos = self._cartesian_from_orbit(
                task.theta, task.radius, task.phi)
            for i in range(self.num_agents):
                if not self._is_alive(i):
                    continue
                if float(np.linalg.norm(self.positions[i] - task_pos)) <= discover_dist and not self.known_tasks[i, task_idx]:
                    self.known_tasks[i, task_idx] = True
                    gained += 1.0
            if self.config.task_knowledge_mode == "local_discovery" and self._task_visible_from_ground(task):
                self.station_known_tasks[task_idx] = True
        return gained

    def _task_visible_from_ground(self, task: Task) -> bool:
        for gs_theta in self.ground_thetas:
            if abs(self._angle_delta(task.theta, float(gs_theta))) <= 1.5 * self.config.ground_contact_angle:
                return True
        return False

    def task_is_known(self, task_idx: int) -> bool:
        if task_idx < 0 or task_idx >= len(self.tasks):
            return False
        return bool(self.known_tasks[:, task_idx].any())

    def _has_path_to_ground(self, src: int) -> bool:
        return self._ground_route_distance(src) is not None

    def _ground_route_distance(self, src: int) -> int | None:
        if not self._is_alive(src):
            return None
        if self._direct_ground_contact(src):
            return 0
        visited = {src}
        frontier = [(src, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            for nxt in np.where(self.comm_adj[node])[0]:
                j = int(nxt)
                if j in visited or not self._is_alive(j):
                    continue
                if self._direct_ground_contact(j):
                    return dist + 1
                visited.add(j)
                frontier.append((j, dist + 1))
        return None

    def _ground_route_score(self, i: int) -> float:
        if not self._is_alive(i):
            return -1.0
        route_dist = self._ground_route_distance(i)
        if route_dist is not None:
            return 4.0 - min(3.0, float(route_dist))
        return float(self.comm_adj[i].sum()) / max(1, self.num_agents - 1)

    def _best_relay_neighbor(self, i: int) -> int | None:
        neighbors = np.where(self.comm_adj[i])[0]
        if len(neighbors) == 0:
            return None
        score_i = self._ground_route_score(i)
        candidates = [int(j) for j in neighbors if self._is_alive(
            int(j)) and self._ground_route_score(int(j)) > score_i]
        if not candidates:
            candidates = [int(j) for j in neighbors if self._is_alive(int(j))]
        if not candidates:
            return None
        return max(candidates, key=self._ground_route_score)

    def _apply_energy_cost(self, i: int, action_name: str) -> float:
        cost = float(self.config.energy_costs.get(
            action_name, self.config.energy_costs["idle"]))
        if action_name in {"relay_ground", "relay_sat"}:
            cost *= 1.0 + self.config.high_orbit_comm_cost_scale * \
                self._high_orbit_factor(i)
        self.energy[i] = max(0.0, float(self.energy[i] - cost))
        return cost

    def _wake_malware(self) -> None:
        if self.rng.random() >= self.config.adversarial_rate:
            return
        healthy = np.where((self.compromised_for <= 0)
                           & (self.health > 0.0))[0]
        if len(healthy) == 0:
            return
        idx = int(self.rng.choice(healthy))
        self.compromised_for[idx] = self.config.compromise_duration
        self.malware_awake[idx] = True

    def _maybe_force_action(self, i: int, act: int) -> int:
        self.last_action_forced[i] = False
        if self.compromised_for[i] <= 0:
            return act
        if self.rng.random() >= self.config.malware_forced_action_prob:
            return act
        self.last_action_forced[i] = True
        nearest = self._nearest_debris_cloud(i)
        if nearest is not None and self.rng.random() < 0.55:
            cloud = nearest
            if cloud.radius > float(self.orbit_radius[i]) + 0.05:
                return 4  # UP toward the debris band
            if cloud.radius < float(self.orbit_radius[i]) - 0.05:
                return 3  # DN toward the debris band
            return 7  # drift through the hazard
        if self._is_low_orbit(i) or self.rng.random() < 0.70:
            return 3  # DN
        return 7

    def _nearest_debris_cloud(self, i: int) -> DebrisCloud | None:
        if not self.config.enable_debris or not self.debris_clouds:
            return None
        active = [d for d in self.debris_clouds if d.density > 1e-4]
        if not active:
            return None
        return min(
            active,
            key=lambda d: abs(self._angle_delta(float(
                self.orbit_theta[i]), d.theta)) + abs(float(self.orbit_radius[i]) - d.radius),
        )

    def _drain_malware(self) -> tuple[np.ndarray, np.ndarray]:
        energy_loss = np.zeros((self.num_agents,), dtype=np.float32)
        health_loss = np.zeros((self.num_agents,), dtype=np.float32)
        for i in range(self.num_agents):
            if not self._is_alive(i) or self.compromised_for[i] <= 0:
                continue
            e = min(float(self.energy[i]), self.config.malware_energy_drain)
            h = min(float(self.health[i]), self.config.malware_health_drain)
            self.energy[i] -= e
            self.health[i] -= h
            energy_loss[i] = e
            health_loss[i] = h
        return energy_loss, health_loss

    def step(self, actions: dict[str, int], agent_names: list[str]) -> tuple[dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        n = self.num_agents
        serviced = np.zeros((n,), dtype=np.float32)
        delivered = np.zeros((n,), dtype=np.float32)
        ground_task_intake = np.zeros((n,), dtype=np.float32)
        knowledge = np.zeros((n,), dtype=np.float32)
        energy_spent = np.zeros((n,), dtype=np.float32)
        overflow = np.zeros((n,), dtype=np.float32)
        data_loss = np.zeros((n,), dtype=np.float32)
        health_loss = np.zeros((n,), dtype=np.float32)
        cyber_penalty = np.zeros((n,), dtype=np.float32)
        jam_penalty = np.zeros((n,), dtype=np.float32)
        forced_penalty = np.zeros((n,), dtype=np.float32)
        atmospheric = np.zeros((n,), dtype=np.float32)
        debris_risk = np.zeros((n,), dtype=np.float32)
        collision = np.zeros((n,), dtype=np.float32)
        executed_actions = ["idle"] * n

        self.jammed[:] = False
        self._wake_malware()
        drained_energy, drained_health = self._drain_malware()
        energy_spent += drained_energy
        health_loss += drained_health
        cyber_penalty += (self.compromised_for > 0).astype(np.float32) * 0.1
        knowledge += self._refresh_task_knowledge() / max(1, n)

        for i, name in enumerate(agent_names):
            if not self._is_alive(i):
                continue
            if not self._is_powered(i):
                executed_actions[i] = "idle"
                self._propagate_kepler(i)
                continue

            act = self._maybe_force_action(i, int(actions.get(name, 7)))
            forced_penalty[i] += 1.0 if self.last_action_forced[i] else 0.0
            action_name = self._action_name(act)
            executed_actions[i] = action_name
            energy_spent[i] += self._apply_energy_cost(i, action_name)
            if self.energy[i] <= 0.0 and action_name != "lowpower":
                self._propagate_kepler(i)
                continue

            if action_name in {"relay_ground", "relay_sat"} and self.compromised_for[i] > 0 and self.rng.random() < self.config.malware_jam_prob:
                self.jammed[i] = True
                jam_penalty[i] += 1.0
                cyber_penalty[i] += 0.5
            elif action_name == "orbit_down":
                self._apply_orbit_shift(i, -self.config.orbit_shift_step)
            elif action_name == "orbit_up":
                self._apply_orbit_shift(i, self.config.orbit_shift_step)
            elif action_name == "observe":
                serviced[i], overflow[i] = self._observe_task(i)
            elif action_name == "relay_ground":
                delivered[i], knowledge[i], ground_task_intake[i] = \
                    self._relay_ground(i)
            elif action_name == "relay_sat":
                knowledge[i], data_loss[i] = self._relay_sat(i)
            elif action_name == "cyberscan":
                self.scan_boost[i] = 4
                if self.compromised_for[i] > 0:
                    self.compromised_for[i] = max(
                        0, self.compromised_for[i] - self.config.scan_duration_reduction)
                    if self.rng.random() < self.config.scan_clean_prob:
                        self.compromised_for[i] = 0
            elif action_name == "lowpower":
                if self.config.enable_recharge and self._in_sunlight(i):
                    self.energy[i] = min(self.config.energy_budget, float(
                        self.energy[i] + self.config.recharge_rate))

            self._propagate_kepler(i)

        if self.config.enable_recharge:
            for i in range(n):
                if self._is_alive(i) and self._in_sunlight(i):
                    self.energy[i] = min(self.config.energy_budget, float(
                        self.energy[i] + 0.15 * self.config.recharge_rate))

        self._refresh_satellite_positions()
        self._update_debris_clouds()
        self.update_comm_graph()
        knowledge += self._refresh_task_knowledge() / max(1, n)

        for i in range(n):
            if not self._is_alive(i):
                continue
            if self._is_low_orbit(i):
                loss = min(float(self.health[i]),
                           self.config.atmospheric_health_loss)
                self.health[i] -= loss
                health_loss[i] += loss
                atmospheric[i] += loss
            pc = self._local_pc_estimate(i)
            if executed_actions[i] in {"orbit_down", "orbit_up"}:
                pc *= (1.0 - self.config.debris_mitigation_factor)
            debris_risk[i] = pc
            if self.config.enable_debris and pc >= self.config.pc_alert_threshold and executed_actions[i] not in {"orbit_down", "orbit_up"}:
                if self.rng.random() < self.config.pc_collision_scale * pc:
                    loss = float(self.rng.uniform(self.config.debris_health_loss_min,
                                 self.config.debris_health_loss_max) * max(0.25, pc))
                    loss = min(float(self.health[i]), loss)
                    self.health[i] -= loss
                    health_loss[i] += loss
                    collision[i] = 1.0
                    if self.health[i] <= 0.0:
                        lost = float(self.buffered_data[i])
                        self.energy[i] = 0.0
                        self.buffered_data[i] = 0.0
                        data_loss[i] += lost

        self._destroy_dead_satellites(data_loss)
        self._update_tasks()
        self.compromised_for = np.maximum(0, self.compromised_for - 1)
        self.malware_awake = self.compromised_for > 0
        self.scan_boost = np.maximum(0, self.scan_boost - 1)
        self.last_executed_actions = executed_actions
        self.t += 1

        alive = int((self.health > 0.0).sum())
        isolated = self._isolated_count()
        failures = float(n - alive)
        components = {
            "task": float(serviced.sum()),
            "delivery": float(delivered.sum()),
            "ground_task_intake": float(ground_task_intake.sum()),
            "knowledge": float(knowledge.sum()),
            "energy": float(energy_spent.sum()),
            "overflow": float(overflow.sum()),
            "data_loss": float(data_loss.sum()),
            "health": float(health_loss.sum()),
            "isolation": float(isolated),
            "failure": failures,
            "cyber": float(cyber_penalty.sum()),
            "jam": float(jam_penalty.sum()),
            "forced_action": float(forced_penalty.sum()),
            "atmospheric_drag": float(atmospheric.sum()),
            "debris_risk": float(debris_risk.sum()),
            "collision": float(collision.sum()),
        }
        self.observed_total += float(serviced.sum())
        self.knowledge_shared_total += float(knowledge.sum())
        self.jam_count += int(jam_penalty.sum())
        self.forced_action_count += int(forced_penalty.sum())
        self.last_reward_components = components
        shared = compute_shared_reward(components, self.config.reward_weights)
        self.last_reward = shared

        rewards = {}
        for i, name in enumerate(agent_names):
            if self.config.reward_mode == "local":
                local_components = {
                    "task": float(serviced[i]),
                    "delivery": float(delivered[i]),
                    "ground_task_intake": float(ground_task_intake[i]),
                    "knowledge": float(knowledge[i]),
                    "energy": float(energy_spent[i]),
                    "overflow": float(overflow[i]),
                    "data_loss": float(data_loss[i]),
                    "health": float(health_loss[i]),
                    "isolation": float(1.0 if self._is_alive(i) and self.comm_adj[i].sum() == 0 else 0.0),
                    "failure": float(0.0 if self._is_alive(i) else 1.0),
                    "cyber": float(cyber_penalty[i]),
                    "jam": float(jam_penalty[i]),
                    "forced_action": float(forced_penalty[i]),
                    "atmospheric_drag": float(atmospheric[i]),
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
                "delivered_total": self.delivered_total,
                "observed_total": self.observed_total,
                "knowledge_shared": self.knowledge_shared_total,
                "energy_mean": float(self.energy.mean()),
                "health_mean": float(self.health.mean()),
                "crashes": int((self.health <= 0.0).sum()),
                "jam_count": int(self.jam_count),
                "forced_actions": int(self.forced_action_count),
                "last_reward": shared,
                "alive": alive,
            }
            for info in infos.values():
                info["episode"] = episode
        return rewards, terminations, truncations, infos

    def _observe_task(self, i: int) -> tuple[float, float]:
        sat_pos = self.positions[i]
        sensing_dist = self._sensing_distance_threshold()
        for task_idx, task in enumerate(self.tasks):
            if not task.active or not self.known_tasks[i, task_idx]:
                continue
            task_pos = self._cartesian_from_orbit(
                task.theta, task.radius, task.phi)
            if float(np.linalg.norm(sat_pos - task_pos)) <= sensing_dist:
                gain = min(self.config.obs_data_gain * task.priority, max(0.0,
                           self.config.data_capacity - float(self.buffered_data[i])))
                overflow = max(0.0, self.config.obs_data_gain *
                               task.priority - gain)
                self.buffered_data[i] += gain
                task.active = False
                self.known_tasks[:, task_idx] = False
                self.station_known_tasks[task_idx] = False
                return float(gain), float(overflow)
        return 0.0, 0.0

    def _relay_ground(self, i: int) -> tuple[float, float, float]:
        if self.jammed[i] or not self._direct_ground_contact(i):
            return 0.0, 0.0, 0.0
        before = int(self.known_tasks[i].sum())
        station_before = int(self.station_known_tasks.sum())
        self.known_tasks[i] |= self.station_known_tasks
        self.station_known_tasks |= self.known_tasks[i]
        ground_task_intake = float(int(self.known_tasks[i].sum()) - before)
        knowledge_gain = ground_task_intake + \
            float(int(self.station_known_tasks.sum()) - station_before)
        out = min(self.config.relay_capacity_ground,
                  float(self.buffered_data[i]))
        self.buffered_data[i] -= out
        self.delivered_total += out
        return float(out), knowledge_gain, ground_task_intake

    def _relay_sat(self, i: int) -> tuple[float, float]:
        if self.jammed[i]:
            return 0.0, 0.0
        j = self._best_relay_neighbor(i)
        if j is None:
            return 0.0, 0.0
        before_i = int(self.known_tasks[i].sum())
        before_j = int(self.known_tasks[j].sum())
        merged = self.known_tasks[i] | self.known_tasks[j]
        self.known_tasks[i] = merged
        self.known_tasks[j] = merged
        knowledge_gain = float((int(self.known_tasks[i].sum(
        )) - before_i) + (int(self.known_tasks[j].sum()) - before_j))
        moved = 0.0
        if self.buffered_data[i] > 0.0 and self._ground_route_score(j) >= self._ground_route_score(i):
            room = max(0.0, self.config.data_capacity -
                       float(self.buffered_data[j]))
            moved = min(self.config.relay_capacity_sat,
                        float(self.buffered_data[i]), room)
            self.buffered_data[i] -= moved
            self.buffered_data[j] += moved
        return knowledge_gain + moved * 0.1, 0.0

    def _destroy_dead_satellites(self, data_loss: np.ndarray) -> None:
        for i in range(self.num_agents):
            if self.health[i] > 0.0:
                continue
            if self.buffered_data[i] > 0.0:
                data_loss[i] += float(self.buffered_data[i])
            self.buffered_data[i] = 0.0
            self.energy[i] = 0.0
            self.known_tasks[i, :] = False
            self.compromised_for[i] = 0
            self.malware_awake[i] = False
            self.jammed[i] = False

    def _update_tasks(self) -> None:
        for idx, task in enumerate(self.tasks):
            if task.active:
                self._propagate_body(task)
                task.age += 1
                if task.age > 25:
                    task.active = False
                    self.known_tasks[:, idx] = False
                    self.station_known_tasks[idx] = False
                elif self.config.task_priority_mode == "dynamic":
                    task.priority = float(
                        np.clip(task.priority + self.rng.uniform(-0.05, 0.08), 0.1, 1.0))
            elif self.rng.random() < self.config.task_spawn_rate:
                self.tasks[idx] = self._spawn_task()
                self.known_tasks[:, idx] = False
                self.station_known_tasks[idx] = self.config.task_knowledge_mode == "ground_catalog"

    def _isolated_count(self) -> int:
        cnt = 0
        for i in range(self.num_agents):
            if self._is_alive(i) and self.comm_adj[i].sum() == 0:
                cnt += 1
        return cnt

    def _mission_failed(self) -> bool:
        alive = int((self.health > 0.0).sum())
        if alive == 0:
            return True
        if alive <= max(1, self.num_agents // 4):
            return True
        return False

    def observe(self, i: int) -> np.ndarray:
        energy_norm = float(
            self.energy[i] / max(self.config.energy_budget, 1e-6))
        health_norm = float(
            self.health[i] / max(self.config.health_budget, 1e-6))
        theta_norm = float(self.orbit_theta[i]) / (2.0 * np.pi)
        radius_norm = (float(self.orbit_radius[i]) - self.config.orbit_min_radius) / max(
            1e-6, self.config.orbit_max_radius - self.config.orbit_min_radius
        )
        radius_norm = float(np.clip(radius_norm, -1.0, 1.0))
        phi_norm = (
            (float(self.orbit_phi[i]) + self.config.inclination_max)
            / max(1e-6, 2.0 * self.config.inclination_max)
            if self.config.world_dim == 3
            else 0.0
        )
        sunlight = 1.0 if self._in_sunlight(i) else 0.0
        ground_contact = 1.0 if self._direct_ground_contact(i) else 0.0
        route_ground = 1.0 if self._has_path_to_ground(i) else 0.0
        deg_norm = float(self.comm_adj[i].sum()) / max(1, self.num_agents - 1)
        buffer_norm = min(
            float(self.buffered_data[i]) / max(self.config.data_capacity, 1e-6), 1.0)
        capacity_remaining = 1.0 - buffer_norm
        known_count, known_prio = self._known_local_task_pressure(i)
        local_debris_density = min(self._local_debris_density(i), 1.0)
        local_pc = self._local_pc_estimate(i)
        compromised = 1.0 if self.compromised_for[i] > 0 else 0.0
        suspicious = 1.0 if self.scan_boost[i] > 0 else 0.0
        jammed = 1.0 if self.jammed[i] else 0.0
        forced = 1.0 if self.last_action_forced[i] else 0.0
        neighbors = np.where(self.comm_adj[i])[0]
        compromised_neighbors = 0.0
        if len(neighbors) > 0:
            compromised_neighbors = float(
                (self.compromised_for[neighbors] > 0).sum()) / len(neighbors)
        alive_frac = float((self.health > 0.0).sum()) / self.num_agents

        obs = np.array([
            energy_norm,
            health_norm,
            theta_norm,
            radius_norm,
            phi_norm,
            sunlight,
            ground_contact,
            route_ground,
            deg_norm,
            buffer_norm,
            capacity_remaining,
            known_count,
            known_prio,
            local_debris_density,
            local_pc,
            compromised,
            suspicious,
            jammed,
            compromised_neighbors,
            alive_frac if not forced else -alive_frac,
        ], dtype=np.float32)

        if self.compromised_for[i] > 0 and self.config.spoof_mode == "obs_spoof":
            obs = obs.copy()
            obs[0] = float(self.rng.uniform(0.0, 1.0))
            obs[2] = float(self.rng.uniform(0.0, 1.0))
            obs[6:13] = self.rng.uniform(0.0, 1.0, size=(7,))
            obs[13:15] = self.rng.uniform(0.0, 1.0, size=(2,))
        return obs

    def _known_local_task_pressure(self, i: int) -> tuple[float, float]:
        local_count = 0.0
        local_prio = 0.0
        sensing_dist = self._discovery_distance_threshold()
        sat_pos = self.positions[i]
        for task_idx, task in enumerate(self.tasks):
            if not task.active or not self.known_tasks[i, task_idx]:
                continue
            task_pos = self._cartesian_from_orbit(
                task.theta, task.radius, task.phi)
            if float(np.linalg.norm(sat_pos - task_pos)) <= sensing_dist:
                local_count += 1.0
                local_prio += task.priority
        local_count = min(local_count / max(1.0, self.config.num_tasks), 1.0)
        local_prio = min(local_prio / max(1.0, self.config.num_tasks), 1.0)
        return float(local_count), float(local_prio)

    def _build_info(self, i: int) -> dict[str, Any]:
        return {
            "energy": float(self.energy[i]),
            "health": float(self.health[i]),
            "alive": bool(self._is_alive(i)),
            "compromised": bool(self.compromised_for[i] > 0),
            "malware_awake": bool(self.malware_awake[i]),
            "jammed": bool(self.jammed[i]),
            "last_action_forced": bool(self.last_action_forced[i]),
            "local_degree": int(self.comm_adj[i].sum()),
            "buffered_data": float(self.buffered_data[i]),
            "known_tasks": int(self.known_tasks[i].sum()),
            "theta": float(self.orbit_theta[i]),
            "phi": float(self.orbit_phi[i]),
            "radius": float(self.orbit_radius[i]),
            "semi_major_axis": float(self.orbit_semi_major_axis[i]),
            "eccentricity": float(self.orbit_eccentricity[i]),
            "mean_anomaly": float(self.orbit_mean_anomaly[i]),
            "arg_periapsis": float(self.orbit_arg_periapsis[i]),
            "inclination": float(self.orbit_inclination[i]),
            "raan": float(self.orbit_raan[i]),
            "sunlight": bool(self._in_sunlight(i)),
            "ground_contact": bool(self._direct_ground_contact(i)),
            "ground_route": bool(self._has_path_to_ground(i)),
            "local_debris_density": float(self._local_debris_density(i)),
            "local_pc_estimate": float(self._local_pc_estimate(i)),
            "last_executed_action": self.last_executed_actions[i],
            "delivered_total": float(self.delivered_total),
            "observed_total": float(self.observed_total),
            "knowledge_shared": float(self.knowledge_shared_total),
            "jam_count": int(self.jam_count),
            "forced_actions": int(self.forced_action_count),
            "time": self.t,
            "reward_components": dict(self.last_reward_components),
        }
