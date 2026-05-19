"""Public ORBITAL environment entrypoints."""

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


def __getattr__(name: str):
    if name in {"OrbitalAECEnv", "env"}:
        from orbital.envs.orbital_aec import OrbitalAECEnv, env

        return OrbitalAECEnv if name == "OrbitalAECEnv" else env
    if name in {"OrbitalParallelEnv", "parallel_env"}:
        from orbital.envs.orbital_parallel import OrbitalParallelEnv, parallel_env

        return OrbitalParallelEnv if name == "OrbitalParallelEnv" else parallel_env
    if name in {"Orbital3DAECEnv", "env3d"}:
        from orbital.envs.orbital3d_aec import Orbital3DAECEnv, env as env3d

        return Orbital3DAECEnv if name == "Orbital3DAECEnv" else env3d
    if name in {"Orbital3DParallelEnv", "parallel_env3d"}:
        from orbital.envs.orbital3d_parallel import Orbital3DParallelEnv, parallel_env as parallel_env3d

        return Orbital3DParallelEnv if name == "Orbital3DParallelEnv" else parallel_env3d
    raise AttributeError(f"module 'orbital.envs' has no attribute {name!r}")
