from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from marl_lib.checkpoint import PolicyCheckpoint
from marl_lib.config import MAPPOConfig
from marl_lib.envs import make_env
from marl_lib.networks import MLPActorCritic
from marl_lib.plotting import generate_training_figures
from marl_lib.preprocessing import preprocess_observation


class MAPPOTrainer:
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
        self.env = make_env(config.env_id, config.env_kwargs, config.organization_path, mma_mode="train", seed=config.seed)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        obs, _infos = self.env.reset(seed=config.seed)
        self.agent_order = list(getattr(self.env, "possible_agents", list(obs.keys())))
        sample_agent = self.agent_order[0]
        sample_obs = self._preprocess_obs(obs[sample_agent])
        self.obs_dim = int(sample_obs.shape[0])
        self.n_agents = len(self.agent_order)
        self.action_dim = int(self.env.action_space(sample_agent).n)
        self.model = MLPActorCritic(self.obs_dim, self.obs_dim * self.n_agents, self.action_dim, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.global_step = 0
        self.best_eval_reward = -float("inf")
        self.update_index = 0
        self.training_history: list[dict[str, Any]] = []
        self.last_rollout_step_stats: list[dict[str, Any]] = []

    def close(self) -> None:
        self.env.close()

    def train(self) -> dict[str, Any]:
        Path(self.config.run_dir).mkdir(parents=True, exist_ok=True)
        metrics: dict[str, Any] = {}
        try:
            while self.global_step < self.config.total_steps:
                t0 = time.time()
                batch, rollout_metrics = self.collect_rollout()
                update_metrics = self.update(batch)
                self.update_index += 1
                metrics = {
                    **rollout_metrics,
                    **update_metrics,
                    "global_step": self.global_step,
                    "total_steps": self.config.total_steps,
                    "update": self.update_index,
                    "seconds": float(time.time() - t0),
                    "steps_per_second": float(len(batch["rewards"]) / max(1e-9, time.time() - t0)),
                }
                should_eval = self.config.eval_on_update or self.global_step % self.config.eval_interval < self.config.rollout_steps
                if should_eval:
                    from marl_lib.eval import EvalRunner

                    eval_metrics = EvalRunner(self.config, self.model, self.agent_order).run()
                    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                    mean_reward = float(eval_metrics["mean_reward"])
                    is_best = mean_reward > self.best_eval_reward
                    if is_best:
                        self.best_eval_reward = mean_reward
                    metrics["best_eval_reward"] = self.best_eval_reward
                else:
                    is_best = False
                self._record_training_stats(metrics)
                self._print_update(metrics)
                self.save_checkpoint("last.pt", metrics)
                if is_best:
                    self.save_checkpoint("best.pt", metrics)
                if self.global_step % self.config.checkpoint_interval < self.config.rollout_steps:
                    self.save_checkpoint(f"step_{self.global_step}.pt", metrics)
                if self.config.target_reward is not None and metrics.get("eval_mean_reward", -float("inf")) >= self.config.target_reward:
                    break
        except KeyboardInterrupt:
            self.save_checkpoint("interrupt.pt", {"global_step": self.global_step, "interrupted": True})
        finally:
            self.close()
        return metrics

    def collect_rollout(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        cfg = self.config
        obs, infos = self.env.reset(seed=cfg.seed + self.global_step)
        rows = []
        episode_rewards = []
        episode_lengths = []
        step_stats = []
        running_reward = 0.0
        running_length = 0
        for _ in range(cfg.rollout_steps):
            agents = list(getattr(self.env, "agents", []))
            if not agents:
                obs, infos = self.env.reset(seed=cfg.seed + self.global_step)
                agents = list(getattr(self.env, "agents", []))
            joint = self._joint_obs(obs)
            actions: dict[str, int] = {}
            logps: dict[str, float] = {}
            values: dict[str, float] = {}
            for agent in agents:
                obs_t = torch.as_tensor(self._preprocess_obs(obs[agent]), dtype=torch.float32, device=self.device).unsqueeze(0)
                joint_t = torch.as_tensor(joint, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model.action_logits(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                actions[agent] = int(action.item())
                logps[agent] = float(dist.log_prob(action).item())
                values[agent] = float(self.model.value(joint_t).item())
            next_obs, rewards, terms, truncs, infos = self.env.step(actions)
            done = bool(any(terms.values()) or any(truncs.values()))
            mean_step_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            step_stats.append(
                {
                    "global_step": int(self.global_step + len(agents)),
                    "mean_reward": mean_step_reward,
                    "sum_reward": float(np.sum(list(rewards.values()))) if rewards else 0.0,
                    "num_agents": int(len(agents)),
                }
            )
            for agent in agents:
                rows.append(
                    {
                        "obs": self._preprocess_obs(obs[agent]),
                        "joint_obs": joint,
                        "action": actions[agent],
                        "logp": logps[agent],
                        "value": values[agent],
                        "reward": float(rewards[agent]),
                        "done": float(done),
                    }
                )
                running_reward += float(rewards[agent]) / max(1, len(agents))
            running_length += 1
            self.global_step += len(agents)
            obs = next_obs
            if done:
                episode_rewards.append(running_reward)
                episode_lengths.append(running_length)
                running_reward = 0.0
                running_length = 0
                obs, infos = self.env.reset(seed=cfg.seed + self.global_step)
        if running_reward:
            episode_rewards.append(running_reward)
            episode_lengths.append(running_length)
        batch = self._make_batch(rows)
        self.last_rollout_step_stats = step_stats
        return batch, {
            "rollout_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "rollout_reward_std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "rollout_reward_var": float(np.var(episode_rewards)) if episode_rewards else 0.0,
            "rollout_episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "train_step_reward_mean": float(np.mean([s["mean_reward"] for s in step_stats])) if step_stats else 0.0,
            "train_step_reward_std": float(np.std([s["mean_reward"] for s in step_stats])) if step_stats else 0.0,
        }

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.config
        adv = self._advantages(batch["rewards"], batch["values"], batch["dones"])
        returns = adv + batch["values"]
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = batch["obs"].shape[0]
        idx = np.arange(n)
        last_policy = last_value = last_entropy = 0.0
        approx_kls = []
        clip_fracs = []
        for _ in range(cfg.update_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                logits = self.model.action_logits(batch["obs"][mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(batch["actions"][mb])
                ratio = torch.exp(new_logp - batch["logps"][mb])
                logratio = new_logp - batch["logps"][mb]
                pg1 = -adv[mb] * ratio
                pg2 = -adv[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value = self.model.value(batch["joint_obs"][mb])
                value_loss = 0.5 * (returns[mb] - value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kls.append(float(((ratio - 1.0) - logratio).mean().item()))
                    clip_fracs.append(float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()))
                last_policy = float(policy_loss.item())
                last_value = float(value_loss.item())
                last_entropy = float(entropy.item())
        explained_var = self._explained_variance(returns.detach(), batch["values"].detach())
        return {
            "policy_loss": last_policy,
            "value_loss": last_value,
            "entropy": last_entropy,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": explained_var,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def save_checkpoint(self, name: str, metrics: dict[str, Any]) -> None:
        PolicyCheckpoint.save(
            Path(self.config.run_dir) / name,
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "global_step": self.global_step,
                "update": self.update_index,
                "best_eval_reward": self.best_eval_reward,
                "metrics": metrics,
                "training_history": self.training_history,
                "agent_order": self.agent_order,
            },
        )
        generate_training_figures(self.training_history, Path(self.config.run_dir) / "figures")

    def load_checkpoint(self, path: str | Path) -> None:
        payload = PolicyCheckpoint.load(path, map_location=str(self.device))
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.update_index = int(payload.get("update", 0))
        self.best_eval_reward = float(payload.get("best_eval_reward", -float("inf")))
        self.training_history = list(payload.get("training_history", []))

    def _joint_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        zero = np.zeros((self.obs_dim,), dtype=np.float32)
        parts = [self._preprocess_obs(obs[agent]) if agent in obs else zero for agent in self.agent_order]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _preprocess_obs(self, obs) -> np.ndarray:
        return preprocess_observation(
            obs,
            image_downsample=self.config.image_downsample,
            image_grayscale=self.config.image_grayscale,
        )

    def _make_batch(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(np.stack([r["obs"] for r in rows]), dtype=torch.float32, device=self.device),
            "joint_obs": torch.as_tensor(np.stack([r["joint_obs"] for r in rows]), dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor([r["action"] for r in rows], dtype=torch.long, device=self.device),
            "logps": torch.as_tensor([r["logp"] for r in rows], dtype=torch.float32, device=self.device),
            "values": torch.as_tensor([r["value"] for r in rows], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor([r["reward"] for r in rows], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor([r["done"] for r in rows], dtype=torch.float32, device=self.device),
        }

    def _advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        adv = torch.zeros_like(rewards)
        lastgaelam = torch.tensor(0.0, device=self.device)
        next_value = torch.tensor(0.0, device=self.device)
        for t in reversed(range(rewards.shape[0])):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        return adv

    def _record_training_stats(self, metrics: dict[str, Any]) -> None:
        row = {k: self._jsonable(v) for k, v in metrics.items()}
        self.training_history.append(row)
        run_dir = Path(self.config.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "training_stats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        with (run_dir / "training_step_rewards.jsonl").open("a", encoding="utf-8") as f:
            for step_row in self.last_rollout_step_stats:
                step_payload = dict(step_row)
                step_payload["update"] = self.update_index
                f.write(json.dumps(step_payload, sort_keys=True) + "\n")
        self._write_training_csv(run_dir / "training_stats.csv")

    def _write_training_csv(self, path: Path) -> None:
        if not self.training_history:
            return
        keys = sorted({key for row in self.training_history for key, value in row.items() if isinstance(value, (int, float, str, bool)) or value is None})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.training_history:
                writer.writerow({key: row.get(key) for key in keys})

    def _print_update(self, metrics: dict[str, Any]) -> None:
        eval_mean = metrics.get("eval_mean_reward", float("nan"))
        eval_var = metrics.get("eval_var_reward", float("nan"))
        msg = (
            f"[MAPPO] update={self.update_index} "
            f"steps={self.global_step}/{self.config.total_steps} "
            f"eval_return_mean={float(eval_mean):.4f} "
            f"eval_return_var={float(eval_var):.4f} "
            f"rollout_return_mean={float(metrics.get('rollout_reward_mean', 0.0)):.4f} "
            f"policy_loss={float(metrics.get('policy_loss', 0.0)):.4f} "
            f"value_loss={float(metrics.get('value_loss', 0.0)):.4f} "
            f"entropy={float(metrics.get('entropy', 0.0)):.4f} "
            f"kl={float(metrics.get('approx_kl', 0.0)):.5f} "
            f"clip_frac={float(metrics.get('clip_fraction', 0.0)):.4f} "
            f"sps={float(metrics.get('steps_per_second', 0.0)):.1f}"
        )
        print(msg, flush=True)

    def _explained_variance(self, returns: torch.Tensor, values: torch.Tensor) -> float:
        y_true = returns.detach().cpu().numpy()
        y_pred = values.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        if var_y < 1e-12:
            return 0.0
        return float(1.0 - np.var(y_true - y_pred) / var_y)

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return value
