from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_lib.checkpoint import PolicyCheckpoint
from marl_lib.config import MAPPOConfig
from marl_lib.eval import EvalRunner
from marl_lib.envs import make_env
from marl_lib.networks import MLPActorCritic
from marl_lib.preprocessing import preprocess_observation
from marl_lib.trainer import MAPPOTrainer
from mma import Organization


def build_empty_organization() -> Organization:
    return Organization(
        name="pistonball_unconstrained_empty",
        roles={},
        goals={},
        assignments={},
        metadata={
            "baseline": "unconstrained_mappo_mma_pistonball",
            "description": "Empty MMA organization: no roles, no goals. MAPPO is unconstrained MARL.",
        },
    )


def build_config(args: argparse.Namespace, org_path: Path) -> MAPPOConfig:
    return MAPPOConfig(
        env_id="pettingzoo.butterfly.pistonball_v6.parallel_env",
        env_kwargs={
            "continuous": False,
            "max_cycles": args.max_cycles,
            "render_mode": None,
        },
        organization_path=str(org_path),
        run_dir=str(args.run_dir),
        seed=args.seed,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        device=args.device,
        image_downsample=args.image_downsample,
        image_grayscale=True,
    )


def load_model_from_checkpoint(checkpoint_path: Path) -> tuple[MAPPOConfig, MLPActorCritic, list[str]]:
    payload = PolicyCheckpoint.load(checkpoint_path)
    cfg = MAPPOConfig.from_dict(payload["config"])
    sample_env = make_env(cfg.env_id, cfg.env_kwargs, cfg.organization_path, mma_mode="eval", seed=cfg.seed)
    try:
        obs, _ = sample_env.reset(seed=cfg.seed)
        agent_order = list(payload.get("agent_order", sample_env.possible_agents))
        sample_agent = agent_order[0]
        obs_dim = len(
            preprocess_observation(
                obs[sample_agent],
                image_downsample=cfg.image_downsample,
                image_grayscale=cfg.image_grayscale,
            )
        )
        action_dim = sample_env.action_space(sample_agent).n
    finally:
        sample_env.close()
    model = MLPActorCritic(obs_dim, obs_dim * len(agent_order), action_dim, cfg.hidden_size)
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, model, agent_order


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    org_path = run_dir / "organization_empty.json"
    stats_path = run_dir / "eval_stats.json"
    gif_path = run_dir / "eval_mosaic.gif"
    run_dir.mkdir(parents=True, exist_ok=True)
    build_empty_organization().to_json(org_path)

    cfg = build_config(args, org_path)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not args.eval_only:
        trainer = MAPPOTrainer(cfg)
        try:
            if args.resume and (run_dir / "last.pt").exists():
                trainer.load_checkpoint(run_dir / "last.pt")
            train_metrics = trainer.train()
        finally:
            trainer.close()
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "last.pt"
    else:
        train_metrics = {"eval_only": True}
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    eval_cfg, model, agent_order = load_model_from_checkpoint(checkpoint_path)
    eval_cfg.eval_episodes = args.eval_episodes
    metrics = EvalRunner(eval_cfg, model, agent_order).run(
        episodes=args.eval_episodes,
        gif_path=str(gif_path),
        deterministic=not args.stochastic_eval,
    )
    payload = {
        "baseline": "unconstrained_mappo_mma_pistonball",
        "environment": "pettingzoo.butterfly.pistonball_v6",
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
    parser = argparse.ArgumentParser(
        description="Unconstrained MAPPO baseline for PettingZoo Pistonball using an empty MMA organization."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baselines/unconstrained_mappo_mma_pistonball"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-cycles", type=int, default=125)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--rollout-steps", type=int, default=500)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--image-downsample", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_baseline(parse_args())
    print(json.dumps(result["eval_metrics"], indent=2, sort_keys=True))
    print(f"checkpoint: {result['checkpoint']}")
    print(f"gif: {result['gif']}")
    print(f"stats: {Path(result['config']['run_dir']) / 'eval_stats.json'}")


if __name__ == "__main__":
    main()
