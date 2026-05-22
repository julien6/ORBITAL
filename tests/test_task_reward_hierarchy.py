import pytest

from orbital.envs.core.config import OrbitalConfig
from orbital.envs.core.dynamics import OrbitalCore
from orbital.envs.core.reward import compute_shared_reward


def test_reward_includes_ground_task_intake_component():
    reward = compute_shared_reward(
        {"ground_task_intake": 2.0},
        {"ground_task_intake": 0.8},
    )

    assert reward == pytest.approx(1.6)


def test_relay_ground_rewards_ground_catalog_task_intake():
    config = OrbitalConfig(
        num_satellites=1,
        num_tasks=3,
        task_spawn_rate=0.0,
        adversarial_rate=0.0,
        enable_debris=False,
        num_debris_clouds=0,
    )
    core = OrbitalCore(config)
    core.reset(seed=7)
    core._direct_ground_contact = lambda _: True

    core.step({"sat_0": 1}, ["sat_0"])

    assert core.last_reward_components["ground_task_intake"] == pytest.approx(3.0)
    assert core.last_reward_components["knowledge"] == pytest.approx(3.0)


def test_default_reward_weights_follow_exponential_hierarchy():
    config = OrbitalConfig()
    weights = config.reward_weights

    assert weights["ground_task_intake"] < weights["knowledge"]
    assert weights["knowledge"] < weights["task"]
    assert weights["task"] < weights["delivery"]

    assert weights["knowledge"] >= weights["ground_task_intake"] * 5
    assert weights["task"] >= weights["knowledge"] * 5
    assert weights["delivery"] >= weights["task"] * 5
