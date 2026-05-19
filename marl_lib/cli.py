from __future__ import annotations

import argparse

from marl_lib.checkpoint import PolicyCheckpoint
from marl_lib.config import MAPPOConfig
from marl_lib.eval import EvalRunner
from marl_lib.networks import MLPActorCritic
from marl_lib.preprocessing import preprocess_observation
from marl_lib.trainer import MAPPOTrainer


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    trainer = MAPPOTrainer(MAPPOConfig.from_file(args.config))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gif")
    args = parser.parse_args(argv)
    payload = PolicyCheckpoint.load(args.checkpoint)
    cfg = MAPPOConfig.from_dict(payload["config"])
    import torch

    sample_env = __import__("marl_lib.envs", fromlist=["make_env"]).make_env(cfg.env_id, cfg.env_kwargs, cfg.organization_path, "eval", cfg.seed)
    obs, _ = sample_env.reset(seed=cfg.seed)
    sample_agent = sample_env.possible_agents[0]
    obs_dim = len(preprocess_observation(
        obs[sample_agent],
        image_downsample=cfg.image_downsample,
        image_grayscale=cfg.image_grayscale,
    ))
    action_dim = sample_env.action_space(sample_agent).n
    agent_order = payload.get("agent_order", list(sample_env.possible_agents))
    sample_env.close()
    model = MLPActorCritic(obs_dim, obs_dim * len(agent_order), action_dim, cfg.hidden_size)
    model.load_state_dict(payload["model"])
    model.eval()
    metrics = EvalRunner(cfg, model, agent_order).run(episodes=args.episodes, gif_path=args.gif)
    print(metrics)


def tune_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune MAPPO hyperparameters with Optuna.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--storage", default="sqlite:///runs/optuna/orbital_mappo.db")
    args = parser.parse_args(argv)
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install optuna or `pip install -e '.[train]'` to tune MAPPO") from exc

    base = MAPPOConfig.from_file(args.config)

    def optuna_goal(trial):
        cfg = MAPPOConfig.from_dict(base.to_dict())
        cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg.gamma = trial.suggest_float("gamma", 0.94, 0.999)
        cfg.gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
        cfg.clip_coef = trial.suggest_float("clip_coef", 0.1, 0.3)
        cfg.entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.05)
        cfg.value_coef = trial.suggest_float("value_coef", 0.25, 1.0)
        cfg.rollout_steps = trial.suggest_categorical("rollout_steps", [64, 128, 256])
        cfg.minibatch_size = trial.suggest_categorical("minibatch_size", [64, 128, 256])
        cfg.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        cfg.run_dir = f"{base.run_dir}/trial_{trial.number}"
        metrics = MAPPOTrainer(cfg).train()
        return float(metrics.get("eval_mean_reward", metrics.get("rollout_reward_mean", 0.0)))

    study = optuna.create_study(direction="maximize", storage=args.storage, study_name="orbital_mappo", load_if_exists=True)
    study.optimize(optuna_goal, n_trials=args.trials)
    print(study.best_params)
