from __future__ import annotations

import numpy as np


class PygameRenderer:
    def __init__(self, width: int = 700, height: int = 700):
        self.width = width
        self.height = height
        self._pygame = None
        self.screen = None
        self.clock = None

    def _ensure(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        self._pygame = pygame
        pygame.init()

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

        self.screen.fill((10, 12, 30))
        margin = 50
        gs = core.config.grid_size
        cell = (self.width - 2 * margin) / max(1, gs - 1)

        if show_links:
            for i in range(core.num_agents):
                for j in range(i + 1, core.num_agents):
                    if core.comm_adj[i, j]:
                        p1 = (margin + core.positions[i, 0] * cell, margin + core.positions[i, 1] * cell)
                        p2 = (margin + core.positions[j, 0] * cell, margin + core.positions[j, 1] * cell)
                        pygame.draw.line(self.screen, (80, 90, 150), p1, p2, 1)

        for t in core.tasks:
            if not t.active:
                continue
            p = (int(margin + t.x * cell), int(margin + t.y * cell))
            intensity = int(120 + 135 * t.priority)
            pygame.draw.circle(self.screen, (intensity, intensity, 80), p, 6)

        gp = (int(margin + core.ground[0] * cell), int(margin + core.ground[1] * cell))
        pygame.draw.rect(self.screen, (120, 220, 120), (gp[0] - 9, gp[1] - 9, 18, 18))

        for i in range(core.num_agents):
            x = int(margin + core.positions[i, 0] * cell)
            y = int(margin + core.positions[i, 1] * cell)
            compromised = core.compromised_for[i] > 0
            color = (90, 180, 255) if core.energy[i] > 0 else (70, 70, 70)
            pygame.draw.circle(self.screen, color, (x, y), 10)
            if compromised:
                pygame.draw.circle(self.screen, (255, 60, 60), (x, y), 13, 2)
            e = core.energy[i] / max(core.config.energy_budget, 1e-6)
            pygame.draw.rect(self.screen, (40, 40, 40), (x - 12, y + 14, 24, 4))
            pygame.draw.rect(self.screen, (80, 220, 100), (x - 12, y + 14, int(24 * max(0.0, min(1.0, e))), 4))

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
