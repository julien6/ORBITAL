import numpy as np

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.dynamics import OrbitalCore


def _angle_delta(after, before):
    return ((np.asarray(after) - np.asarray(before) + np.pi) % (2.0 * np.pi)) - np.pi


def test_orbiting_bodies_advance_mean_anomaly_from_semi_major_axis():
    config = OrbitalConfig(adversarial_rate=0.0, p_link_drop=0.0, task_spawn_rate=0.0)
    core = OrbitalCore(config)
    core.reset(seed=42)

    satellite_mean_anomaly = core.orbit_mean_anomaly.copy()
    satellite_axis = core.orbit_semi_major_axis.copy()
    task_mean_anomaly = np.array([task.orbit.mean_anomaly for task in core.tasks])
    task_axis = np.array([task.orbit.semi_major_axis for task in core.tasks])
    debris_mean_anomaly = np.array([cloud.orbit.mean_anomaly for cloud in core.debris_clouds])
    debris_axis = np.array([cloud.orbit.semi_major_axis for cloud in core.debris_clouds])

    actions = {f"sat_{idx}": 7 for idx in range(core.num_agents)}
    core.step(actions, list(actions))

    np.testing.assert_allclose(
        _angle_delta(core.orbit_mean_anomaly, satellite_mean_anomaly),
        config.kepler_constant / satellite_axis**1.5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        _angle_delta([task.orbit.mean_anomaly for task in core.tasks], task_mean_anomaly),
        config.kepler_constant / task_axis**1.5,
    )
    np.testing.assert_allclose(
        _angle_delta([cloud.orbit.mean_anomaly for cloud in core.debris_clouds], debris_mean_anomaly),
        config.kepler_constant / debris_axis**1.5,
    )


def test_default_satellite_orbits_are_elliptic_and_change_radius():
    core = OrbitalCore(OrbitalConfig(adversarial_rate=0.0, task_spawn_rate=0.0))
    core.reset(seed=7)

    first_radius = core.orbit_radius.copy()
    actions = {f"sat_{idx}": 7 for idx in range(core.num_agents)}
    for _ in range(5):
        core.step(actions, list(actions))

    assert np.all(core.orbit_eccentricity > 0.0)
    assert np.any(np.abs(core.orbit_radius - first_radius) > 1e-4)
