"""ORBITAL environment package."""

__all__ = ["env", "parallel_env", "env3d", "parallel_env3d"]


def __getattr__(name: str):
    if name == "env":
        from orbital.envs.orbital_aec import env

        return env
    if name == "parallel_env":
        from orbital.envs.orbital_parallel import parallel_env

        return parallel_env
    if name == "env3d":
        from orbital.envs.orbital3d_aec import env as env3d

        return env3d
    if name == "parallel_env3d":
        from orbital.envs.orbital3d_parallel import parallel_env as parallel_env3d

        return parallel_env3d
    raise AttributeError(f"module 'orbital' has no attribute {name!r}")
