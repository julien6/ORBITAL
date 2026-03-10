"""Public ORBITAL environment entrypoints.

`env()` exposes the PettingZoo AEC API.
`parallel_env()` exposes the PettingZoo Parallel API.
Both share the same core dynamics and configuration schema.
"""

from orbital.envs.orbital_aec import OrbitalAECEnv, env
from orbital.envs.orbital_parallel import OrbitalParallelEnv, parallel_env
from orbital.envs.orbital3d_aec import Orbital3DAECEnv, env as env3d
from orbital.envs.orbital3d_parallel import Orbital3DParallelEnv, parallel_env as parallel_env3d

__all__ = [
    "OrbitalAECEnv",
    "OrbitalParallelEnv",
    "Orbital3DAECEnv",
    "Orbital3DParallelEnv",
    "env",
    "parallel_env",
    "env3d",
    "parallel_env3d",
]
