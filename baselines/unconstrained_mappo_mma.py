from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanmarl.orbital_runner import CleanMAPPOConfig, evaluate_policy, load_checkpoint, train_mappo
from mma import Organization


def build_empty_organization() -> Organization:
    return Organization(
        name="unconstrained_empty",
        roles={},
        goals={},
        assignments={},
        metadata={
            "baseline": "unconstrained_cleanmarl_mappo_mma",
            "description": "Empty MMA organization: no roles, no goals. CleanMARL MAPPO is unconstrained.",
        },
    )


def build_config(args: argparse.Namespace, org_path: Path) -> CleanMAPPOConfig:
    return CleanMAPPOConfig(
        env_factory="orbital.envs.orbital_parallel.parallel_env",
        env_kwargs={
            "num_satellites": args.satellites,
            "max_steps": args.max_steps,
            "render_mode": None,
            "render_projection": "2d",
            "enable_debris": True,
        },
        organization_path=str(org_path),
        run_dir=str(args.run_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        epochs=args.epochs,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        eval_steps=args.eval_steps,
        num_eval_ep=args.eval_episodes,
        device=args.device,
    )


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    org_path = run_dir / "organization_empty.json"
    stats_path = run_dir / "eval_stats.json"
    gif_path = run_dir / "eval_mosaic.gif"
    build_empty_organization().to_json(org_path)
    cfg = build_config(args, org_path)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"

    if not args.eval_only:
        train_metrics = train_mappo(cfg)
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "last.pt"
    else:
        train_metrics = {"eval_only": True}
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    eval_cfg, actor, _critic, _payload = load_checkpoint(checkpoint_path, device=args.device)
    metrics = evaluate_policy(eval_cfg, actor, episodes=args.eval_episodes, gif_path=str(gif_path))
    payload = {
        "baseline": "unconstrained_cleanmarl_mappo_mma",
        "organization": str(org_path),
        "checkpoint": str(checkpoint_path),
        "gif": str(gif_path),
        "train_metrics": train_metrics,
        "eval_metrics": metrics,
        "config": cfg.to_dict(),
    }
    stats_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unconstrained CleanMARL MAPPO baseline for ORBITAL.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baselines/unconstrained_cleanmarl_mappo_mma"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--satellites", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--actor-hidden-dim", type=int, default=32)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate-actor", type=float, default=8e-4)
    parser.add_argument("--learning-rate-critic", type=float, default=8e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    result = run_baseline(parse_args())
    print(json.dumps(result["eval_metrics"], indent=2, sort_keys=True))
    print(f"checkpoint: {result['checkpoint']}")
    print(f"gif: {result['gif']}")
    print(f"stats: {Path(result['config']['run_dir']) / 'eval_stats.json'}")


if __name__ == "__main__":
    main()
