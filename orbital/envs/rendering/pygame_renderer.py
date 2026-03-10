from __future__ import annotations

import numpy as np


class PygameRenderer:
    def __init__(self, width: int = 980, height: int = 740):
        self.width = width
        self.height = height
        self._pygame = None
        self.screen = None
        self.clock = None
        self._fonts = {}
        self._trail_by_agent = {}
        self._trail_len = 20
        self._last_t = -1
        self._orbit_x_scale = 1.0
        self._orbit_y_scale = 0.82
        self._anim_prev_orbit = {}
        self._anim_target_orbit = {}
        self._anim_current_orbit = {}
        self._anim_progress = 1.0
        self._anim_duration_ms = 220.0
        self._anim_core_t = -1
        self._last_tick_ms = None

    def _ensure(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.font.init()

    def _font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = self._pygame.font.SysFont("DejaVu Sans", size, bold=bold)
        return self._fonts[key]

    def _draw_text(self, text: str, x: int, y: int, color, size: int = 14, bold: bool = False):
        surf = self._font(size, bold=bold).render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _role_tag(self, core, i: int) -> str:
        energy_ratio = core.energy[i] / max(core.config.energy_budget, 1e-6)
        if core.compromised_for[i] > 0 or energy_ratio < 0.25:
            return "SAFE"
        if core.buffered_data[i] > 1.0 or core._has_path_to_ground(i):
            return "REL"
        return "OBS"

    def _task_color(self, priority: float):
        p = float(np.clip(priority, 0.0, 1.0))
        low = np.array([225, 216, 95], dtype=np.float32)
        high = np.array([250, 106, 52], dtype=np.float32)
        col = (1.0 - p) * low + p * high
        return tuple(int(v) for v in col)

    def _model_orbit_to_visual(self, theta: float, radius: float, r_src_min: float, r_src_max: float, r_vis_min: float, r_vis_max: float) -> tuple[float, float]:
        rn = (float(radius) - float(r_src_min)) / max(1e-6, float(r_src_max - r_src_min))
        rn = float(np.clip(rn, 0.0, 1.0))
        r_vis = float(r_vis_min + rn * (r_vis_max - r_vis_min))
        return float(theta), r_vis

    def _orbit_to_point(self, theta: float, r: float, cx: int, cy: int) -> tuple[int, int]:
        return (
            int(cx + self._orbit_x_scale * r * np.cos(theta)),
            int(cy + self._orbit_y_scale * r * np.sin(theta)),
        )

    def _orbital_point(self, theta: float, radius: float, cx: int, cy: int, r_src_min: float, r_src_max: float, r_vis_min: float, r_vis_max: float) -> tuple[int, int]:
        theta, r = self._model_orbit_to_visual(
            theta=theta,
            radius=radius,
            r_src_min=r_src_min,
            r_src_max=r_src_max,
            r_vis_min=r_vis_min,
            r_vis_max=r_vis_max,
        )
        return self._orbit_to_point(theta=theta, r=r, cx=cx, cy=cy)

    def _project_3d_to_map(self, x: float, y: float, z: float, cx: int, cy: int, r_vis: float) -> tuple[int, int]:
        # Oblique view for 3D mode while preserving a compact 2D mission panel.
        yaw = np.deg2rad(35.0)
        pitch = np.deg2rad(28.0)
        u = (x * np.cos(yaw)) - (y * np.sin(yaw))
        v = ((x * np.sin(yaw)) + (y * np.cos(yaw))) * np.cos(pitch) - z * np.sin(pitch)
        norm = float(np.hypot(u, v))
        if norm < 1e-9:
            return (cx, cy)
        return (
            int(cx + self._orbit_x_scale * r_vis * (u / norm)),
            int(cy + self._orbit_y_scale * r_vis * (v / norm)),
        )

    def _wrap_angle_delta(self, a0: float, a1: float) -> float:
        # shortest angular displacement in [-pi, pi]
        return float(((a1 - a0 + np.pi) % (2.0 * np.pi)) - np.pi)

    def _draw_dashed_line(self, p1: tuple[int, int], p2: tuple[int, int], color, width: int = 1, dash: int = 7, gap: int = 4):
        pygame = self._pygame
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            return
        ux = dx / dist
        uy = dy / dist
        s = 0.0
        while s < dist:
            e = min(s + dash, dist)
            a = (int(x1 + ux * s), int(y1 + uy * s))
            b = (int(x1 + ux * e), int(y1 + uy * e))
            pygame.draw.line(self.screen, color, a, b, width)
            s += dash + gap

    def _segment_circle_intersections(self, p1: tuple[int, int], p2: tuple[int, int], cx: int, cy: int, radius: float) -> list[float]:
        x1 = float(p1[0] - cx)
        y1 = float(p1[1] - cy)
        x2 = float(p2[0] - cx)
        y2 = float(p2[1] - cy)
        dx = x2 - x1
        dy = y2 - y1
        a = dx * dx + dy * dy
        if a < 1e-9:
            return []
        b = 2.0 * (x1 * dx + y1 * dy)
        c = x1 * x1 + y1 * y1 - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []
        sq = float(np.sqrt(max(0.0, disc)))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        ts = []
        if 0.0 <= t1 <= 1.0:
            ts.append(float(t1))
        if 0.0 <= t2 <= 1.0:
            ts.append(float(t2))
        ts.sort()
        return ts

    def _clip_line_to_earth(self, p1: tuple[int, int], p2: tuple[int, int], cx: int, cy: int, earth_r: float) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        ts = self._segment_circle_intersections(p1, p2, cx, cy, earth_r)
        if not ts:
            return [(p1, p2)]
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        points = [p1]
        for t in ts:
            points.append((int(x1 + t * dx), int(y1 + t * dy)))
        points.append(p2)
        segments = []
        for a, b in zip(points[:-1], points[1:]):
            mx = 0.5 * (a[0] + b[0]) - cx
            my = 0.5 * (a[1] + b[1]) - cy
            if (mx * mx + my * my) >= (earth_r * earth_r):
                segments.append((a, b))
        return segments

    def _ray_until_earth(self, p_from: tuple[int, int], p_to: tuple[int, int], cx: int, cy: int, earth_r: float) -> tuple[int, int]:
        ts = self._segment_circle_intersections(p_from, p_to, cx, cy, earth_r)
        if not ts:
            return p_to
        t = ts[0]
        x1, y1 = float(p_from[0]), float(p_from[1])
        x2, y2 = float(p_to[0]), float(p_to[1])
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        return (x, y)

    def render(self, core, mode: str = "human", show_links: bool = True):
        self._ensure()
        pygame = self._pygame
        if mode == "human":
            if self.screen is None:
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
        else:
            if self.screen is None:
                self.screen = pygame.Surface((self.width, self.height))

        bg = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            frac = y / max(1, self.height - 1)
            c0 = np.array([8, 11, 26], dtype=np.float32)
            c1 = np.array([18, 24, 52], dtype=np.float32)
            c = (1.0 - frac) * c0 + frac * c1
            pygame.draw.line(bg, tuple(int(v) for v in c), (0, y), (self.width, y))
        self.screen.blit(bg, (0, 0))
        now_ms = pygame.time.get_ticks()
        if self._last_tick_ms is None:
            self._last_tick_ms = now_ms
        dt_ms = float(max(0, now_ms - self._last_tick_ms))
        self._last_tick_ms = now_ms

        map_margin = 28
        panel_gap = 20
        panel_w = 300
        map_size = min(self.height - 2 * map_margin, self.width - panel_w - 2 * map_margin - panel_gap)
        map_rect = pygame.Rect(map_margin, map_margin, map_size, map_size)
        panel_rect = pygame.Rect(map_rect.right + panel_gap, map_margin, panel_w, map_size)

        panel_bg = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
        panel_bg.fill((18, 23, 46, 220))
        self.screen.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(self.screen, (68, 85, 125), panel_rect, 1, border_radius=10)

        grid_bg = pygame.Surface((map_rect.w, map_rect.h), pygame.SRCALPHA)
        grid_bg.fill((12, 18, 42, 220))
        self.screen.blit(grid_bg, map_rect.topleft)
        pygame.draw.rect(self.screen, (70, 92, 138), map_rect, 1, border_radius=8)

        cx = map_rect.left + map_rect.w // 2
        cy = map_rect.top + map_rect.h // 2
        max_orbit = int(0.46 * map_rect.w)
        earth_r = int(0.16 * map_rect.w)
        leo_r1 = int(earth_r + 0.24 * (max_orbit - earth_r))
        leo_r2 = int(earth_r + 0.56 * (max_orbit - earth_r))
        leo_r3 = int(earth_r + 0.88 * (max_orbit - earth_r))

        # faint star field
        for idx in range(90):
            sx = int(map_rect.left + ((idx * 73) % map_rect.w))
            sy = int(map_rect.top + ((idx * 41 + 17) % map_rect.h))
            s = 1 + (idx % 2)
            a = 40 + (idx % 4) * 30
            star = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
            pygame.draw.circle(star, (170, 190, 255, a), (s, s), s)
            self.screen.blit(star, (sx, sy))

        # LEO shells
        pygame.draw.circle(self.screen, (70, 92, 138), (cx, cy), leo_r1, 1)
        pygame.draw.circle(self.screen, (78, 104, 152), (cx, cy), leo_r2, 1)
        pygame.draw.circle(self.screen, (88, 118, 170), (cx, cy), leo_r3, 1)
        self._draw_text("LEO-1", cx + 8, cy - leo_r1 - 16, (140, 176, 240), size=11)
        self._draw_text("LEO-2", cx + 8, cy - leo_r2 - 16, (140, 176, 240), size=11)
        self._draw_text("LEO-3", cx + 8, cy - leo_r3 - 16, (140, 176, 240), size=11)

        r_src_min = float(core.config.orbit_min_radius)
        r_src_max = float(core.config.orbit_max_radius)
        r_span_src = max(1e-6, r_src_max - r_src_min)
        r_scale = (float(leo_r3) - float(leo_r1)) / r_span_src
        earth_r = int(max(10.0, float(leo_r1) - r_scale * (r_src_min - float(core.config.earth_radius))))

        # Earth
        earth = pygame.Surface((earth_r * 2 + 4, earth_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(earth, (58, 114, 196), (earth_r + 2, earth_r + 2), earth_r)
        pygame.draw.circle(earth, (82, 158, 230), (earth_r - 18, earth_r - 12), int(0.52 * earth_r))
        pygame.draw.circle(earth, (72, 152, 102), (earth_r + 12, earth_r - 6), int(0.36 * earth_r))
        pygame.draw.circle(earth, (72, 152, 102), (earth_r - 8, earth_r + 22), int(0.28 * earth_r))
        self.screen.blit(earth, (cx - earth_r - 2, cy - earth_r - 2))
        pygame.draw.circle(self.screen, (145, 182, 238), (cx, cy), earth_r, 2)
        self._draw_text("EARTH", cx - 26, cy - 8, (222, 238, 255), size=12, bold=True)

        def map_px(theta: float, radius: float, phi: float = 0.0) -> tuple[int, int]:
            if core.config.world_dim == 3:
                _, r_vis = self._model_orbit_to_visual(
                    theta=float(theta),
                    radius=float(radius),
                    r_src_min=r_src_min,
                    r_src_max=r_src_max,
                    r_vis_min=float(leo_r1),
                    r_vis_max=float(leo_r3),
                )
                cphi = np.cos(float(phi))
                x = np.cos(float(theta)) * cphi
                y = np.sin(float(theta)) * cphi
                z = np.sin(float(phi))
                return self._project_3d_to_map(x, y, z, cx, cy, r_vis)
            return self._orbital_point(
                theta=float(theta),
                radius=float(radius),
                cx=cx,
                cy=cy,
                r_src_min=r_src_min,
                r_src_max=r_src_max,
                r_vis_min=float(leo_r1),
                r_vis_max=float(leo_r3),
            )

        # Keep short orbital trails to make motion and orbiting easier to read.
        if core.t <= self._last_t:
            self._trail_by_agent = {}
            self._anim_prev_orbit = {}
            self._anim_target_orbit = {}
            self._anim_current_orbit = {}
            self._anim_progress = 1.0
            self._anim_core_t = -1
        sat_points = []
        if core.t != self._anim_core_t:
            new_target = {}
            for i in range(core.num_agents):
                if core.config.world_dim == 3:
                    new_target[i] = map_px(
                        theta=float(core.orbit_theta[i]),
                        radius=float(core.orbit_radius[i]),
                        phi=float(core.orbit_phi[i]),
                    )
                else:
                    new_target[i] = self._model_orbit_to_visual(
                        theta=float(core.orbit_theta[i]),
                        radius=float(core.orbit_radius[i]),
                        r_src_min=r_src_min,
                        r_src_max=r_src_max,
                        r_vis_min=float(leo_r1),
                        r_vis_max=float(leo_r3),
                    )
            if not self._anim_current_orbit:
                self._anim_current_orbit = dict(new_target)
            self._anim_prev_orbit = dict(self._anim_current_orbit)
            self._anim_target_orbit = dict(new_target)
            for i in range(core.num_agents):
                if i not in self._anim_prev_orbit:
                    self._anim_prev_orbit[i] = self._anim_target_orbit[i]
            self._anim_progress = 1.0 if mode != "human" else 0.0
            self._anim_core_t = core.t
        elif mode == "human" and self._anim_progress < 1.0:
            self._anim_progress = min(1.0, self._anim_progress + dt_ms / max(1.0, self._anim_duration_ms))

        alpha = 1.0 if mode != "human" else float(np.clip(self._anim_progress, 0.0, 1.0))
        self._anim_current_orbit = {}
        for i in range(core.num_agents):
            if core.config.world_dim == 3:
                fallback_x, fallback_y = map_px(
                    theta=float(core.orbit_theta[i]),
                    radius=float(core.orbit_radius[i]),
                    phi=float(core.orbit_phi[i]),
                )
                t_x, t_y = self._anim_target_orbit.get(i, (fallback_x, fallback_y))
                p_x, p_y = self._anim_prev_orbit.get(i, (t_x, t_y))
                cur_x = int(p_x + alpha * (t_x - p_x))
                cur_y = int(p_y + alpha * (t_y - p_y))
                self._anim_current_orbit[i] = (cur_x, cur_y)
                p = (cur_x, cur_y)
            else:
                fallback_theta, fallback_r = self._model_orbit_to_visual(
                    theta=float(core.orbit_theta[i]),
                    radius=float(core.orbit_radius[i]),
                    r_src_min=r_src_min,
                    r_src_max=r_src_max,
                    r_vis_min=float(leo_r1),
                    r_vis_max=float(leo_r3),
                )
                t_theta, t_r = self._anim_target_orbit.get(i, (fallback_theta, fallback_r))
                p_theta, p_r = self._anim_prev_orbit.get(i, (t_theta, t_r))
                d_theta = self._wrap_angle_delta(p_theta, t_theta)
                cur_theta = p_theta + alpha * d_theta
                cur_r = p_r + alpha * (t_r - p_r)
                self._anim_current_orbit[i] = (cur_theta, cur_r)
                p = self._orbit_to_point(theta=cur_theta, r=cur_r, cx=cx, cy=cy)
            sat_points.append(p)

            tr = self._trail_by_agent.setdefault(i, [])
            if not tr or tr[-1] != p:
                tr.append(p)
            if len(tr) > self._trail_len:
                del tr[: len(tr) - self._trail_len]
        self._last_t = core.t

        for i in range(core.num_agents):
            tr = self._trail_by_agent.get(i, [])
            if len(tr) < 2:
                continue
            for k in range(1, len(tr)):
                fade = 0.25 + 0.75 * (k / len(tr))
                col = (int(70 * fade), int(130 * fade), int(220 * fade))
                pygame.draw.line(self.screen, col, tr[k - 1], tr[k], 1)

        if show_links:
            for i in range(core.num_agents):
                for j in range(i + 1, core.num_agents):
                    if core.comm_adj[i, j]:
                        p1 = sat_points[i]
                        p2 = sat_points[j]
                        highlighted = core._has_path_to_ground(i) or core._has_path_to_ground(j)
                        link_color = (112, 196, 255) if highlighted else (78, 96, 148)
                        for a, b in self._clip_line_to_earth(p1, p2, cx, cy, float(earth_r)):
                            pygame.draw.line(self.screen, link_color, a, b, 2 if highlighted else 1)

        # Debris clouds are visualized as translucent hazard halos.
        for d in getattr(core, "debris_clouds", []):
            if d.density <= 1e-4:
                continue
            px, py = map_px(d.theta, d.radius, d.phi)
            rr = int(6 + 32 * float(np.clip(d.spread / max(1e-3, core.config.debris_spread_max), 0.0, 1.0)))
            alpha = int(40 + 120 * float(np.clip(d.density, 0.0, 1.0)))
            halo = pygame.Surface((2 * rr + 2, 2 * rr + 2), pygame.SRCALPHA)
            pygame.draw.circle(halo, (255, 120, 70, alpha), (rr + 1, rr + 1), rr)
            pygame.draw.circle(halo, (255, 165, 90, min(255, alpha + 40)), (rr + 1, rr + 1), max(3, rr // 2), 1)
            self.screen.blit(halo, (px - rr - 1, py - rr - 1))

        for t in core.tasks:
            if not t.active:
                continue
            p = map_px(t.theta, t.radius, t.phi)
            rad = 4 + int(5 * float(np.clip(t.priority, 0.0, 1.0)))
            pygame.draw.circle(self.screen, self._task_color(t.priority), p, rad)
            age_alpha = max(30, 140 - 4 * t.age)
            age_ring = pygame.Surface((2 * (rad + 3), 2 * (rad + 3)), pygame.SRCALPHA)
            pygame.draw.circle(age_ring, (250, 170, 70, age_alpha), (rad + 3, rad + 3), rad + 2, 1)
            self.screen.blit(age_ring, (p[0] - (rad + 3), p[1] - (rad + 3)))

        # Ground stations are shown on the Earth limb and act as downlink anchors.
        ground_points = []
        downlink_r = int(0.52 * (leo_r1 - earth_r) + 22)
        for gi, (g_theta, g_phi) in enumerate(zip(core.ground_thetas, core.ground_phis)):
            if core.config.world_dim == 3:
                gx, gy = map_px(float(g_theta), float(core.config.earth_radius), float(g_phi))
            else:
                gx = int(cx + (earth_r + 2) * np.cos(float(g_theta)))
                gy = int(cy + (earth_r + 2) * np.sin(float(g_theta)))
            ground_points.append((gx, gy))
            pygame.draw.circle(self.screen, (72, 188, 122), (gx, gy), downlink_r, 1)
            pygame.draw.rect(self.screen, (124, 236, 150), (gx - 7, gy - 7, 14, 14), border_radius=2)
            self._draw_text(f"GS{gi}", gx + 10, gy - 10, (175, 244, 176), size=11, bold=True)

        for i in range(core.num_agents):
            x, y = sat_points[i]
            compromised = core.compromised_for[i] > 0
            alive = core.energy[i] > 0
            isolated = alive and int(core.comm_adj[i].sum()) == 0
            role = self._role_tag(core, i)
            energy_ratio = core.energy[i] / max(core.config.energy_budget, 1e-6)
            color = (90, 180, 255) if alive else (70, 70, 70)
            if role == "REL" and alive:
                color = (115, 214, 255)
            elif role == "SAFE" and alive:
                color = (144, 226, 175)

            if isolated:
                pygame.draw.circle(self.screen, (255, 195, 90), (x, y), 16, 2)
            pygame.draw.circle(self.screen, color, (x, y), 10)
            if compromised:
                pulse = 13 + ((core.t + i) % 2)
                pygame.draw.circle(self.screen, (255, 74, 74), (x, y), pulse, 2)

            if core.buffered_data[i] > 0.0 and alive:
                b = float(np.clip(core.buffered_data[i] / 3.0, 0.0, 1.0))
                pygame.draw.rect(self.screen, (38, 48, 88), (x - 12, y - 20, 24, 4), border_radius=2)
                pygame.draw.rect(self.screen, (116, 228, 255), (x - 12, y - 20, int(24 * b), 4), border_radius=2)
                can_downlink = core._direct_ground_contact(i) or core._has_path_to_ground(i)
                if core.config.world_dim == 3:
                    sat_vec = core.positions[i]
                    nearest_g_idx = int(np.argmin([float(np.linalg.norm(sat_vec - core.ground_vectors[j])) for j in range(len(core.ground_vectors))]))
                else:
                    sat_theta = float(core.orbit_theta[i])
                    nearest_g_idx = int(np.argmin([abs(core._angle_delta(sat_theta, float(gt))) for gt in core.ground_thetas]))
                gx, gy = ground_points[nearest_g_idx]
                target = (gx, gy)
                if not can_downlink:
                    target = self._ray_until_earth((x, y), (gx, gy), cx, cy, float(earth_r))
                if can_downlink:
                    for a, b2 in self._clip_line_to_earth((x, y), target, cx, cy, float(earth_r)):
                        pygame.draw.line(self.screen, (102, 238, 165), a, b2, 1)
                else:
                    for a, b2 in self._clip_line_to_earth((x, y), target, cx, cy, float(earth_r)):
                        self._draw_dashed_line(a, b2, (244, 132, 132), width=1, dash=5, gap=4)

            if alive and core._direct_ground_contact(i):
                pygame.draw.circle(self.screen, (88, 242, 154), (x + 11, y - 10), 3)

            e = energy_ratio
            pygame.draw.rect(self.screen, (40, 40, 40), (x - 12, y + 14, 24, 4))
            pygame.draw.rect(self.screen, (80, 220, 100), (x - 12, y + 14, int(24 * max(0.0, min(1.0, e))), 4))
            self._draw_text(str(i), x - 4, y - 35, (215, 228, 255), size=12, bold=True)
            self._draw_text(role, x - 14, y + 21, (160, 224, 244), size=10, bold=True)

        active_tasks = sum(1 for t in core.tasks if t.active)
        active_debris = sum(1 for d in getattr(core, "debris_clouds", []) if d.density > 1e-4)
        alive_count = int((core.energy > 0).sum())
        isolated_count = sum(1 for i in range(core.num_agents) if core.energy[i] > 0 and int(core.comm_adj[i].sum()) == 0)
        y = panel_rect.top + 16
        x = panel_rect.left + 14

        self._draw_text("ORBITAL MISSION VIEW", x, y, (226, 234, 255), size=18, bold=True)
        y += 34
        self._draw_text(f"t = {core.t}/{core.config.max_steps}", x, y, (190, 205, 240))
        y += 26
        self._draw_text(f"Alive: {alive_count}/{core.num_agents}", x, y, (173, 240, 188))
        y += 20
        self._draw_text(f"Active tasks: {active_tasks}", x, y, (246, 223, 132))
        y += 20
        self._draw_text(f"Isolated sats: {isolated_count}", x, y, (255, 197, 106))
        y += 20
        self._draw_text(f"Active debris clouds: {active_debris}", x, y, (255, 154, 124))
        y += 20
        self._draw_text(f"Delivered total: {core.delivered_total:.1f}", x, y, (124, 226, 255))

        y += 30
        self._draw_text("Last Reward Components", x, y, (218, 231, 255), size=14, bold=True)
        y += 22
        rc = core.last_reward_components
        self._draw_text(f"task: {rc.get('task', 0.0):>5.2f}", x, y, (245, 227, 130))
        y += 18
        self._draw_text(f"delivery: {rc.get('delivery', 0.0):>5.2f}", x, y, (128, 229, 252))
        y += 18
        self._draw_text(f"energy: -{rc.get('energy', 0.0):>4.2f}", x, y, (255, 181, 108))
        y += 18
        self._draw_text(f"isolation: -{rc.get('isolation', 0.0):>4.2f}", x, y, (255, 191, 118))
        y += 18
        self._draw_text(f"failure: -{rc.get('failure', 0.0):>4.2f}", x, y, (255, 126, 126))
        y += 18
        self._draw_text(f"cyber: -{rc.get('cyber', 0.0):>4.2f}", x, y, (255, 142, 164))
        y += 18
        self._draw_text(f"debris: -{rc.get('debris_risk', 0.0):>4.2f}", x, y, (255, 166, 124))
        y += 18
        self._draw_text(f"collision: -{rc.get('collision', 0.0):>4.2f}", x, y, (255, 98, 82))

        y += 30
        self._draw_text("Legend", x, y, (218, 231, 255), size=14, bold=True)
        y += 22
        pygame.draw.circle(self.screen, (58, 114, 196), (x + 8, y + 8), 6)
        self._draw_text("Earth (center disk)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.circle(self.screen, (88, 118, 170), (x + 8, y + 8), 7, 1)
        self._draw_text("LEO orbital shells", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.circle(self.screen, (244, 196, 89), (x + 8, y + 8), 6)
        self._draw_text("Task (brighter = higher priority)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.rect(self.screen, (124, 236, 150), (x + 2, y + 1, 12, 12), border_radius=2)
        self._draw_text("Ground stations + downlink zones", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.circle(self.screen, (255, 130, 85), (x + 8, y + 8), 6)
        self._draw_text("Orbital debris cloud (hazard)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.line(self.screen, (112, 196, 255), (x + 2, y + 7), (x + 14, y + 7), 2)
        self._draw_text("Comms link (clipped by Earth)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.line(self.screen, (102, 238, 165), (x + 2, y + 7), (x + 14, y + 7), 1)
        self._draw_text("Downlink feasible (buffered sat)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        self._draw_dashed_line((x + 2, y + 7), (x + 14, y + 7), (244, 132, 132), width=1, dash=3, gap=2)
        self._draw_text("Downlink blocked (buffered sat)", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.line(self.screen, (70, 130, 220), (x + 2, y + 7), (x + 14, y + 7), 1)
        self._draw_text("Recent orbital trail", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.circle(self.screen, (255, 74, 74), (x + 8, y + 8), 7, 2)
        self._draw_text("Compromised satellite alert", x + 20, y, (202, 218, 246), size=12)
        y += 18
        pygame.draw.rect(self.screen, (116, 228, 255), (x + 2, y + 5, 12, 4), border_radius=2)
        self._draw_text("Buffered data bar (top of sat)", x + 20, y, (202, 218, 246), size=12)

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
            return None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self.screen = None
        self.clock = None
        self._fonts = {}
        self._trail_by_agent = {}
        self._last_t = -1
        self._last_tick_ms = None
        self._anim_prev_orbit = {}
        self._anim_target_orbit = {}
        self._anim_current_orbit = {}
        self._anim_progress = 1.0
        self._anim_core_t = -1
