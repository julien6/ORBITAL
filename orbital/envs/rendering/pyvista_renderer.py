from __future__ import annotations

from typing import Any

import numpy as np


class PyVistaRenderer:
    """True 3D renderer using PyVista, optimized with persistent VTK buffers."""

    def __init__(self, width: int = 960, height: int = 640):
        self.width = width
        self.height = height
        self._pv = None
        self._plotter_human = None
        self._human_shown = False
        self._scene_initialized = False
        self._render_quality = "medium"
        self._far = 1e6

        # Persistent dynamic buffers (rebuilt only when topology sizes change).
        self._dyn_shape: tuple[int, int, int] | None = None  # (n_agents, n_tasks, n_debris)
        self._sat_poly = None
        self._task_poly = None
        self._debris_poly = None
        self._alert_poly = None
        self._isolated_poly = None
        self._contact_poly = None
        self._links_poly = None
        self._downlink_ok_poly = None
        self._downlink_blocked_poly = None
        self._max_links = 0

    def _ensure_pyvista(self):
        if self._pv is not None:
            return
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "render_projection='3d' requires pyvista. "
                "Install with: pip install -e '.[render3d]'"
            ) from exc
        self._pv = pv

    @staticmethod
    def _to_rgb01(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    @staticmethod
    def _task_color(priority: float) -> tuple[int, int, int]:
        p = float(np.clip(priority, 0.0, 1.0))
        low = np.array([225, 216, 95], dtype=np.float32)
        high = np.array([250, 106, 52], dtype=np.float32)
        col = (1.0 - p) * low + p * high
        return tuple(int(v) for v in col)

    @staticmethod
    def _to_xyz(vec: np.ndarray) -> tuple[float, float, float]:
        if vec.shape[0] == 3:
            return (float(vec[0]), float(vec[1]), float(vec[2]))
        return (float(vec[0]), float(vec[1]), 0.0)

    @staticmethod
    def _quality_params(quality: str) -> dict[str, int]:
        if quality == "ultra_low":
            return {
                "earth_res": 12,
                "ground_res": 8,
                "ring_pts": 36,
                "sat_size": 9,
                "task_size": 6,
                "debris_size": 8,
                "alert_size": 13,
                "link_width": 1,
                "smooth": 0,
            }
        if quality == "low":
            return {
                "earth_res": 18,
                "ground_res": 10,
                "ring_pts": 60,
                "sat_size": 12,
                "task_size": 8,
                "debris_size": 11,
                "alert_size": 18,
                "link_width": 1,
                "smooth": 0,
            }
        if quality == "high":
            return {
                "earth_res": 48,
                "ground_res": 22,
                "ring_pts": 160,
                "sat_size": 20,
                "task_size": 13,
                "debris_size": 20,
                "alert_size": 30,
                "link_width": 3,
                "smooth": 1,
            }
        return {
            "earth_res": 32,
            "ground_res": 16,
            "ring_pts": 100,
            "sat_size": 16,
            "task_size": 11,
            "debris_size": 16,
            "alert_size": 25,
            "link_width": 2,
            "smooth": 1,
        }

    def _ground_xyz(self, core, idx: int) -> tuple[float, float, float]:
        if core.config.world_dim == 3:
            g = core.ground_vectors[idx]
            return (float(g[0]), float(g[1]), float(g[2]))
        theta = float(core.ground_thetas[idx])
        r = float(core.config.earth_radius)
        return (r * float(np.cos(theta)), r * float(np.sin(theta)), 0.0)

    def _leo_shell_radii(self, core) -> list[float]:
        r_min = float(core.config.orbit_min_radius)
        r_max = float(core.config.orbit_max_radius)
        return [
            r_min + 0.24 * (r_max - r_min),
            r_min + 0.56 * (r_max - r_min),
            r_min + 0.88 * (r_max - r_min),
        ]

    def _orbit_ring_polydata(self, radius: float, npts: int):
        pv = self._pv
        ang = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False, dtype=np.float32)
        pts = np.stack(
            [radius * np.cos(ang), radius * np.sin(ang), np.zeros_like(ang)],
            axis=1,
        )
        lines = np.empty((npts, 3), dtype=np.int32)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(npts, dtype=np.int32)
        lines[:, 2] = (np.arange(npts, dtype=np.int32) + 1) % npts
        poly = pv.PolyData(pts)
        poly.lines = lines.reshape(-1)
        return poly

    def _init_dynamic_buffers(self, core) -> None:
        pv = self._pv
        n_agents = core.num_agents
        n_tasks = len(core.tasks)
        n_debris = len(getattr(core, "debris_clouds", []))
        shape = (n_agents, n_tasks, n_debris)
        if self._dyn_shape == shape and self._sat_poly is not None:
            return

        self._dyn_shape = shape
        self._max_links = n_agents * (n_agents - 1) // 2
        far3 = np.array([self._far, self._far, self._far], dtype=np.float32)

        sat_pts = np.tile(far3, (n_agents, 1))
        self._sat_poly = pv.PolyData(sat_pts)
        self._sat_poly["rgb"] = np.zeros((n_agents, 3), dtype=np.uint8)

        task_pts = np.tile(far3, (max(1, n_tasks), 1))
        self._task_poly = pv.PolyData(task_pts)
        self._task_poly["rgb"] = np.zeros((max(1, n_tasks), 3), dtype=np.uint8)

        debris_pts = np.tile(far3, (max(1, n_debris), 1))
        self._debris_poly = pv.PolyData(debris_pts)
        self._debris_poly["rgb"] = np.zeros((max(1, n_debris), 3), dtype=np.uint8)

        alert_pts = np.tile(far3, (max(1, n_agents), 1))
        self._alert_poly = pv.PolyData(alert_pts)
        isolated_pts = np.tile(far3, (max(1, n_agents), 1))
        self._isolated_poly = pv.PolyData(isolated_pts)
        contact_pts = np.tile(far3, (max(1, n_agents), 1))
        self._contact_poly = pv.PolyData(contact_pts)

        # Keep one persistent links actor; update points/lines every frame.
        link_pts = np.tile(far3, (max(2, 2 * self._max_links), 1))
        self._links_poly = pv.PolyData(link_pts)
        self._links_poly.lines = np.array([2, 0, 1], dtype=np.int32)
        down_ok_pts = np.tile(far3, (max(2, 2 * n_agents), 1))
        self._downlink_ok_poly = pv.PolyData(down_ok_pts)
        self._downlink_ok_poly.lines = np.array([2, 0, 1], dtype=np.int32)
        down_blocked_pts = np.tile(far3, (max(2, 2 * n_agents), 1))
        self._downlink_blocked_poly = pv.PolyData(down_blocked_pts)
        self._downlink_blocked_poly.lines = np.array([2, 0, 1], dtype=np.int32)

    def _init_scene(self, plotter: Any, core) -> None:
        pv = self._pv
        q = self._quality_params(self._render_quality)
        earth_r = float(core.config.earth_radius)

        earth = pv.Sphere(radius=earth_r, theta_resolution=q["earth_res"], phi_resolution=q["earth_res"])
        plotter.add_mesh(
            earth,
            color=self._to_rgb01((58, 114, 196)),
            smooth_shading=bool(q["smooth"]),
            specular=0.18,
            name="earth",
        )

        shell_cols = [(70, 92, 138), (78, 104, 152), (88, 118, 170)]
        for idx, (rr, cc) in enumerate(zip(self._leo_shell_radii(core), shell_cols)):
            ring = self._orbit_ring_polydata(rr, q["ring_pts"])
            plotter.add_mesh(ring, color=self._to_rgb01(cc), line_width=2, name=f"leo_ring_{idx}")

        gs_rad = 0.035 * earth_r
        for gi in range(len(core.ground_thetas)):
            gx, gy, gz = self._ground_xyz(core, gi)
            gs = pv.Sphere(
                radius=gs_rad,
                center=(gx, gy, gz),
                theta_resolution=q["ground_res"],
                phi_resolution=q["ground_res"],
            )
            plotter.add_mesh(
                gs,
                color=self._to_rgb01((124, 236, 150)),
                smooth_shading=bool(q["smooth"]),
                name=f"ground_{gi}",
            )

        self._init_dynamic_buffers(core)
        plotter.add_points(
            self._sat_poly,
            scalars="rgb",
            rgb=True,
            render_points_as_spheres=True,
            point_size=q["sat_size"],
            name="sat_points",
        )
        plotter.add_points(
            self._task_poly,
            scalars="rgb",
            rgb=True,
            render_points_as_spheres=True,
            point_size=q["task_size"],
            name="task_points",
        )
        plotter.add_points(
            self._debris_poly,
            scalars="rgb",
            rgb=True,
            render_points_as_spheres=True,
            point_size=q["debris_size"],
            opacity=0.55,
            name="debris_points",
        )
        plotter.add_points(
            self._alert_poly,
            color=self._to_rgb01((255, 74, 74)),
            render_points_as_spheres=True,
            point_size=q["alert_size"],
            opacity=0.25,
            name="sat_alert_points",
        )
        plotter.add_points(
            self._isolated_poly,
            color=self._to_rgb01((255, 195, 90)),
            render_points_as_spheres=True,
            point_size=max(q["alert_size"] - 2, q["sat_size"] + 4),
            opacity=0.24,
            name="sat_isolated_points",
        )
        plotter.add_points(
            self._contact_poly,
            color=self._to_rgb01((88, 242, 154)),
            render_points_as_spheres=True,
            point_size=max(5, q["task_size"] - 1),
            opacity=0.95,
            name="sat_contact_points",
        )
        plotter.add_mesh(
            self._links_poly,
            color=self._to_rgb01((88, 145, 210)),
            line_width=q["link_width"],
            name="links",
        )
        plotter.add_mesh(
            self._downlink_ok_poly,
            color=self._to_rgb01((102, 238, 165)),
            line_width=max(1, q["link_width"]),
            name="downlink_ok",
        )
        plotter.add_mesh(
            self._downlink_blocked_poly,
            color=self._to_rgb01((244, 132, 132)),
            line_width=max(1, q["link_width"]),
            opacity=0.85,
            name="downlink_blocked",
        )

        max_r = float(core.config.orbit_max_radius)
        cdist = 3.0 * max_r
        plotter.camera_position = [(cdist, cdist, 0.9 * cdist), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        self._scene_initialized = True

    def _update_links_buffer(self, core, show_links: bool) -> None:
        n_agents = core.num_agents
        far = np.array([self._far, self._far, self._far], dtype=np.float32)

        if not show_links:
            self._links_poly.points[:] = far
            self._links_poly.lines = np.array([2, 0, 1], dtype=np.int32)
            return

        points = []
        lines = []
        k = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if not core.comm_adj[i, j]:
                    continue
                points.append(self._to_xyz(core.positions[i]))
                points.append(self._to_xyz(core.positions[j]))
                lines.extend([2, 2 * k, 2 * k + 1])
                k += 1

        if k == 0:
            self._links_poly.points[:] = far
            self._links_poly.lines = np.array([2, 0, 1], dtype=np.int32)
            return

        pts_arr = np.asarray(points, dtype=np.float32)
        self._links_poly.points[: pts_arr.shape[0], :] = pts_arr
        if self._links_poly.points.shape[0] > pts_arr.shape[0]:
            self._links_poly.points[pts_arr.shape[0] :, :] = far
        self._links_poly.lines = np.asarray(lines, dtype=np.int32)

    def _update_downlink_buffers(self, core, has_ground_path: list[bool]) -> None:
        n_agents = core.num_agents
        far = np.array([self._far, self._far, self._far], dtype=np.float32)
        ground_pts = np.array([self._ground_xyz(core, gi) for gi in range(len(core.ground_thetas))], dtype=np.float32)
        if ground_pts.shape[0] == 0:
            self._downlink_ok_poly.points[:] = far
            self._downlink_ok_poly.lines = np.array([2, 0, 1], dtype=np.int32)
            self._downlink_blocked_poly.points[:] = far
            self._downlink_blocked_poly.lines = np.array([2, 0, 1], dtype=np.int32)
            return

        ok_points: list[tuple[float, float, float]] = []
        ok_lines: list[int] = []
        blk_points: list[tuple[float, float, float]] = []
        blk_lines: list[int] = []
        ok_k = 0
        blk_k = 0

        for i in range(n_agents):
            if core.energy[i] <= 0 or core.buffered_data[i] <= 0.0:
                continue
            sat = np.asarray(self._to_xyz(core.positions[i]), dtype=np.float32)
            nearest = int(np.argmin(np.linalg.norm(ground_pts - sat[None, :], axis=1)))
            g = ground_pts[nearest]
            can_downlink = core._direct_ground_contact(i) or has_ground_path[i]
            if can_downlink:
                ok_points.append((float(sat[0]), float(sat[1]), float(sat[2])))
                ok_points.append((float(g[0]), float(g[1]), float(g[2])))
                ok_lines.extend([2, 2 * ok_k, 2 * ok_k + 1])
                ok_k += 1
            else:
                blk_points.append((float(sat[0]), float(sat[1]), float(sat[2])))
                blk_points.append((float(g[0]), float(g[1]), float(g[2])))
                blk_lines.extend([2, 2 * blk_k, 2 * blk_k + 1])
                blk_k += 1

        if ok_k > 0:
            ok_pts_arr = np.asarray(ok_points, dtype=np.float32)
            self._downlink_ok_poly.points[: ok_pts_arr.shape[0], :] = ok_pts_arr
            if self._downlink_ok_poly.points.shape[0] > ok_pts_arr.shape[0]:
                self._downlink_ok_poly.points[ok_pts_arr.shape[0] :, :] = far
            self._downlink_ok_poly.lines = np.asarray(ok_lines, dtype=np.int32)
        else:
            self._downlink_ok_poly.points[:] = far
            self._downlink_ok_poly.lines = np.array([2, 0, 1], dtype=np.int32)

        if blk_k > 0:
            blk_pts_arr = np.asarray(blk_points, dtype=np.float32)
            self._downlink_blocked_poly.points[: blk_pts_arr.shape[0], :] = blk_pts_arr
            if self._downlink_blocked_poly.points.shape[0] > blk_pts_arr.shape[0]:
                self._downlink_blocked_poly.points[blk_pts_arr.shape[0] :, :] = far
            self._downlink_blocked_poly.lines = np.asarray(blk_lines, dtype=np.int32)
        else:
            self._downlink_blocked_poly.points[:] = far
            self._downlink_blocked_poly.lines = np.array([2, 0, 1], dtype=np.int32)

    def _update_dynamic_buffers(self, core, show_links: bool) -> None:
        n_agents = core.num_agents
        n_tasks = len(core.tasks)
        n_debris = len(getattr(core, "debris_clouds", []))
        shape = (n_agents, n_tasks, n_debris)
        if self._dyn_shape != shape:
            # Rebuild on topology change.
            self._scene_initialized = False
            return

        far = np.array([self._far, self._far, self._far], dtype=np.float32)

        # Compute once (this used to be repeated in a loop).
        has_ground_path = [core._has_path_to_ground(i) for i in range(n_agents)]

        sat_pts = np.zeros((n_agents, 3), dtype=np.float32)
        sat_cols = np.zeros((n_agents, 3), dtype=np.uint8)
        alert_pts = np.tile(far, (max(1, n_agents), 1))
        isolated_pts = np.tile(far, (max(1, n_agents), 1))
        contact_pts = np.tile(far, (max(1, n_agents), 1))
        alert_idx = 0
        isolated_idx = 0
        contact_idx = 0
        for i in range(n_agents):
            sat_pts[i] = np.asarray(self._to_xyz(core.positions[i]), dtype=np.float32)
            alive = core.energy[i] > 0
            compromised = core.compromised_for[i] > 0
            energy_ratio = core.energy[i] / max(core.config.energy_budget, 1e-6)
            color = (90, 180, 255) if alive else (70, 70, 70)
            if alive and (core.buffered_data[i] > 1.0 or has_ground_path[i]):
                color = (115, 214, 255)
            if alive and (compromised or energy_ratio < 0.25):
                color = (144, 226, 175)
            sat_cols[i] = np.array(color, dtype=np.uint8)
            if compromised and alert_idx < alert_pts.shape[0]:
                alert_pts[alert_idx] = sat_pts[i]
                alert_idx += 1
            if alive and int(core.comm_adj[i].sum()) == 0 and isolated_idx < isolated_pts.shape[0]:
                isolated_pts[isolated_idx] = sat_pts[i]
                isolated_idx += 1
            if alive and core._direct_ground_contact(i) and contact_idx < contact_pts.shape[0]:
                contact_pts[contact_idx] = sat_pts[i]
                contact_idx += 1

        self._sat_poly.points[:] = sat_pts
        self._sat_poly["rgb"][:] = sat_cols
        self._alert_poly.points[:] = alert_pts
        self._isolated_poly.points[:] = isolated_pts
        self._contact_poly.points[:] = contact_pts

        task_pts = np.tile(far, (max(1, n_tasks), 1))
        task_cols = np.zeros((max(1, n_tasks), 3), dtype=np.uint8)
        for ti, t in enumerate(core.tasks):
            if not t.active:
                continue
            task_pts[ti] = np.asarray(
                self._to_xyz(core._cartesian_from_orbit(t.theta, t.radius, t.phi)),
                dtype=np.float32,
            )
            task_cols[ti] = np.array(self._task_color(t.priority), dtype=np.uint8)
        self._task_poly.points[:] = task_pts
        self._task_poly["rgb"][:] = task_cols

        debris_pts = np.tile(far, (max(1, n_debris), 1))
        debris_cols = np.zeros((max(1, n_debris), 3), dtype=np.uint8)
        for di, d in enumerate(getattr(core, "debris_clouds", [])):
            if d.density <= 1e-4:
                continue
            debris_pts[di] = np.asarray(
                self._to_xyz(core._cartesian_from_orbit(d.theta, d.radius, d.phi)),
                dtype=np.float32,
            )
            v = float(np.clip(d.density, 0.0, 1.0))
            debris_cols[di] = np.array((255, int(120 + 45 * v), int(70 + 20 * v)), dtype=np.uint8)
        self._debris_poly.points[:] = debris_pts
        self._debris_poly["rgb"][:] = debris_cols

        self._update_links_buffer(core, show_links=show_links)
        self._update_downlink_buffers(core, has_ground_path=has_ground_path)

    def _update_hud(self, plotter: Any, core) -> None:
        alive = int((core.energy > 0).sum())
        active_tasks = sum(1 for t in core.tasks if t.active)
        active_debris = sum(1 for d in getattr(core, "debris_clouds", []) if d.density > 1e-4)
        rc = core.last_reward_components
        hud = (
            f"ORBITAL 3D VIEW | t={core.t}/{core.config.max_steps}\n"
            f"alive={alive}/{core.num_agents} | tasks={active_tasks} | debris={active_debris}\n"
            f"delivered={core.delivered_total:.1f} | task={rc.get('task', 0.0):.2f} | delivery={rc.get('delivery', 0.0):.2f}\n"
            f"energy=-{rc.get('energy', 0.0):.2f} | isolation=-{rc.get('isolation', 0.0):.2f} | cyber=-{rc.get('cyber', 0.0):.2f}\n"
            f"debris=-{rc.get('debris_risk', 0.0):.2f} | collision=-{rc.get('collision', 0.0):.2f}"
        )
        plotter.add_text(hud, position="upper_left", font_size=10, color="white", name="hud")

    def _configure_plotter(self, plotter: Any) -> None:
        plotter.set_background((8 / 255.0, 11 / 255.0, 26 / 255.0))
        try:
            plotter.disable_anti_aliasing()
        except Exception:
            pass

    def _render_offscreen(self, core, show_links: bool):
        pv = self._pv
        p = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        self._configure_plotter(p)
        self._init_scene(p, core)
        self._update_dynamic_buffers(core, show_links=show_links)
        if self._scene_initialized:
            self._update_hud(p, core)
        img = p.screenshot(return_img=True)
        p.close()
        return np.asarray(img, dtype=np.uint8)

    def render(self, core, mode: str = "human", show_links: bool = True):
        self._ensure_pyvista()
        pv = self._pv
        self._render_quality = getattr(core.config, "render_quality", "medium")

        if mode == "human":
            if self._plotter_human is None:
                self._plotter_human = pv.Plotter(
                    off_screen=False,
                    window_size=[self.width, self.height],
                    title="ORBITAL 3D",
                )
                self._configure_plotter(self._plotter_human)

            p = self._plotter_human
            if not self._scene_initialized:
                self._init_scene(p, core)

            self._update_dynamic_buffers(core, show_links=show_links)
            if not self._scene_initialized:
                # Topology changed; rebuild once.
                p.clear()
                self._configure_plotter(p)
                self._init_scene(p, core)
                self._update_dynamic_buffers(core, show_links=show_links)

            self._update_hud(p, core)

            if not self._human_shown:
                p.show(auto_close=False, interactive_update=True)
                self._human_shown = True
            p.render()
            return None

        if mode == "rgb_array":
            return self._render_offscreen(core, show_links=show_links)

        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        if self._plotter_human is not None:
            self._plotter_human.close()
            self._plotter_human = None
        self._human_shown = False
        self._scene_initialized = False
        self._dyn_shape = None
        self._sat_poly = None
        self._task_poly = None
        self._debris_poly = None
        self._alert_poly = None
        self._isolated_poly = None
        self._contact_poly = None
        self._links_poly = None
        self._downlink_ok_poly = None
        self._downlink_blocked_poly = None
        self._max_links = 0
        self._pv = None
