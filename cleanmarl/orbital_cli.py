from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleanmarl.orbital_runner import CleanMAPPOConfig, evaluate_policy, load_checkpoint, train_mappo


def config_from_args(args) -> CleanMAPPOConfig:
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}
    return CleanMAPPOConfig(
        env_factory=args.env_factory,
        env_kwargs=env_kwargs,
        organization_path=args.organization_path,
        run_dir=args.run_dir,
        seed=args.seed,
        agent_ids=not args.no_agent_ids,
        batch_size=args.batch_size,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        epochs=args.epochs,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        clip_gradients=args.clip_gradients,
        eval_steps=args.eval_steps,
        num_eval_ep=args.num_eval_ep,
        device=args.device,
        image_downsample=args.image_downsample,
        image_grayscale=args.image_grayscale,
    )


def add_common(parser):
    parser.add_argument("--env-factory", default="orbital.envs.orbital_parallel.parallel_env")
    parser.add_argument("--env-kwargs", default='{"num_satellites": 6, "max_steps": 96, "render_mode": null, "render_projection": "2d"}')
    parser.add_argument("--organization-path")
    parser.add_argument("--run-dir", default="runs/cleanmarl_orbital")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-agent-ids", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--actor-hidden-dim", type=int, default=32)
    parser.add_argument("--actor-num-layers", type=int, default=1)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-num-layers", type=int, default=1)
    parser.add_argument("--learning-rate-actor", type=float, default=8e-4)
    parser.add_argument("--learning-rate-critic", type=float, default=8e-4)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--clip-gradients", type=float, default=-1.0)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--num-eval-ep", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-downsample", type=int, default=1)
    parser.add_argument("--image-grayscale", action="store_true")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CleanMARL MAPPO on ORBITAL-compatible envs.")
    add_common(parser)
    args = parser.parse_args(argv)
    metrics = train_mappo(config_from_args(args))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate CleanMARL MAPPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--gif")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    cfg, actor, _critic, _payload = load_checkpoint(args.checkpoint, device=args.device)
    metrics = evaluate_policy(cfg, actor, episodes=args.episodes, gif_path=args.gif)
    stats_path = Path(args.checkpoint).parent / "eval_stats.json"
    stats_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def tune_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for CleanMARL MAPPO.")
    add_common(parser)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--storage", default="sqlite:///runs/optuna/cleanmarl_mappo.db")
    args = parser.parse_args(argv)
    import optuna

    base = config_from_args(args)

    def objective(trial):
        cfg = CleanMAPPOConfig.from_dict(base.to_dict())
        cfg.learning_rate_actor = trial.suggest_float("learning_rate_actor", 1e-5, 2e-3, log=True)
        cfg.learning_rate_critic = trial.suggest_float("learning_rate_critic", 1e-5, 2e-3, log=True)
        cfg.gamma = trial.suggest_float("gamma", 0.94, 0.999)
        cfg.td_lambda = trial.suggest_float("td_lambda", 0.85, 0.99)
        cfg.ppo_clip = trial.suggest_float("ppo_clip", 0.1, 0.3)
        cfg.entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.05)
        cfg.actor_hidden_dim = trial.suggest_categorical("actor_hidden_dim", [32, 64, 128])
        cfg.critic_hidden_dim = trial.suggest_categorical("critic_hidden_dim", [64, 128, 256])
        cfg.batch_size = trial.suggest_categorical("batch_size", [1, 2, 3])
        cfg.epochs = trial.suggest_categorical("epochs", [2, 3, 4])
        cfg.run_dir = str(Path(base.run_dir) / f"trial_{trial.number}")
        metrics = train_mappo(cfg)
        return float(metrics.get("eval_mean_reward", metrics.get("rollout_reward_mean", 0.0)))

    study = optuna.create_study(direction="maximize", storage=args.storage, study_name="cleanmarl_mappo", load_if_exists=True)
    study.optimize(objective, n_trials=args.trials)
    print(study.best_params)


if __name__ == "__main__":
    train_main()
