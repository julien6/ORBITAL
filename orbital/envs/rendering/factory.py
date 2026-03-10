from __future__ import annotations

from orbital.envs.rendering.pygame_renderer import PygameRenderer
from orbital.envs.rendering.pyvista_renderer import PyVistaRenderer


def create_renderer(render_projection: str):
    if render_projection == "3d":
        return PyVistaRenderer()
    return PygameRenderer()

