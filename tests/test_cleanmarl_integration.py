from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

import cleanmarl.mappo as mappo
from cleanmarl.env.orbital_wrapper import ParallelEnvCleanMARLWrapper, preprocess_obs
from cleanmarl.orbital_runner import CleanMAPPOConfig, evaluate_policy, load_checkpoint, train_mappo
from mma import AgentAssignment, Organization, Role, RoleRule
from mma.registry import register_role_rule


def test_cleanmarl_mappo_imports():
    assert mappo.Actor is not None
    assert mappo.Critic is not None


def test_orbital_cleanmarl_wrapper_contract():
    env = ParallelEnvCleanMARLWrapper(
        "orbital.envs.orbital_parallel.parallel_env",
        {"num_satellites": 2, "max_steps": 4, "enable_debris": False, "adversarial_rate": 0.0},
        seed=3,
    )
    try:
        obs, _ = env.reset(seed=3)
        assert env.n_agents == 2
        assert obs.shape[0] == 2
        assert env.get_state().shape[0] == env.get_state_size()
        assert env.get_avail_actions().shape == (2, env.get_action_size())
        next_obs, reward, done, trunc, info = env.step([0, 7])
        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


@register_role_rule("test.cleanmarl.only_idle")
def _only_idle(history, joint, agent, params):
    return [7]


def test_orbital_cleanmarl_wrapper_applies_mma_replacement(tmp_path: Path):
    org = Organization(
        name="only_idle",
        roles={"IdleOnly": Role("IdleOnly", [RoleRule("test.cleanmarl.only_idle")])},
        goals={},
        assignments={"sat_0": AgentAssignment("sat_0", ["IdleOnly"], [])},
    )
    org_path = tmp_path / "org.json"
    org.to_json(org_path)
    env = ParallelEnvCleanMARLWrapper(
        "orbital.envs.orbital_parallel.parallel_env",
        {"num_satellites": 1, "max_steps": 4, "enable_debris": False, "adversarial_rate": 0.0},
        organization_path=str(org_path),
        seed=4,
    )
    try:
        env.reset(seed=4)
        env.step([0])
        assert env.env.history[-1]["executed_actions"]["sat_0"] == 7
    finally:
        env.close()


def test_pistonball_preprocessing_contract():
    import numpy as np

    image = np.full((8, 6, 3), 255, dtype=np.uint8)
    out = preprocess_obs(image, image_downsample=2, image_grayscale=True)
    assert out.shape == (12,)
    assert float(out.max()) == 1.0


def test_cleanmarl_short_training_checkpoint_and_gif(tmp_path: Path):
    org = Organization(name="empty", roles={}, goals={}, assignments={})
    org_path = tmp_path / "empty.json"
    org.to_json(org_path)
    cfg = CleanMAPPOConfig(
        env_factory="orbital.envs.orbital_parallel.parallel_env",
        env_kwargs={"num_satellites": 2, "max_steps": 4, "enable_debris": False, "adversarial_rate": 0.0, "render_mode": None},
        organization_path=str(org_path),
        run_dir=str(tmp_path / "run"),
        seed=5,
        batch_size=1,
        total_timesteps=4,
        epochs=1,
        eval_steps=1,
        num_eval_ep=1,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
    )
    train_mappo(cfg)
    ckpt = tmp_path / "run" / "best.pt"
    assert ckpt.exists()
    eval_cfg, actor, _critic, _payload = load_checkpoint(ckpt)
    gif = tmp_path / "run" / "eval_mosaic.gif"
    metrics = evaluate_policy(eval_cfg, actor, episodes=1, gif_path=gif)
    assert "mean_reward" in metrics
    assert gif.exists()
    frame = imageio.get_reader(gif).get_data(0)
    assert frame.max() > frame.min()
