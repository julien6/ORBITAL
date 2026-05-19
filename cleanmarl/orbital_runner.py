from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanmarl.env.orbital_wrapper import ParallelEnvCleanMARLWrapper
from cleanmarl.mappo import Actor, Critic, RolloutBuffer, norm_d


@dataclass
class CleanMAPPOConfig:
    env_factory: str = "orbital.envs.orbital_parallel.parallel_env"
    env_kwargs: dict[str, Any] | None = None
    organization_path: str | None = None
    run_dir: str = "runs/cleanmarl_orbital"
    seed: int = 1
    agent_ids: bool = True
    batch_size: int = 3
    actor_hidden_dim: int = 32
    actor_num_layers: int = 1
    critic_hidden_dim: int = 64
    critic_num_layers: int = 1
    optimizer: str = "Adam"
    learning_rate_actor: float = 8e-4
    learning_rate_critic: float = 8e-4
    total_timesteps: int = 100000
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_reward: bool = False
    normalize_advantage: bool = False
    normalize_return: bool = False
    epochs: int = 3
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1.0
    eval_steps: int = 10
    num_eval_ep: int = 4
    device: str = "cpu"
    image_downsample: int = 1
    image_grayscale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CleanMAPPOConfig":
        return cls(**data)


def make_env(cfg: CleanMAPPOConfig, mode: str, render_mode: str | None = None):
    env_kwargs = dict(cfg.env_kwargs or {})
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    return ParallelEnvCleanMARLWrapper(
        env_factory=cfg.env_factory,
        env_kwargs=env_kwargs,
        organization_path=cfg.organization_path,
        mma_mode=mode,
        agent_ids=cfg.agent_ids,
        image_downsample=cfg.image_downsample,
        image_grayscale=cfg.image_grayscale,
        seed=cfg.seed,
    )


def train_mappo(cfg: CleanMAPPOConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env(cfg, "train")
    eval_env = make_env(cfg, "eval")
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    actor = Actor(env.get_obs_size(), cfg.actor_hidden_dim, cfg.actor_num_layers, env.get_action_size()).to(device)
    critic = Critic(env.get_state_size(), cfg.critic_hidden_dim, cfg.critic_num_layers).to(device)
    optimizer_cls = getattr(optim, cfg.optimizer)
    actor_optimizer = optimizer_cls(actor.parameters(), lr=cfg.learning_rate_actor)
    critic_optimizer = optimizer_cls(critic.parameters(), lr=cfg.learning_rate_critic)
    rb = RolloutBuffer(
        buffer_size=cfg.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=cfg.normalize_reward,
        device=device,
    )

    history: list[dict[str, Any]] = []
    step = 0
    update = 0
    best_eval = -float("inf")
    metrics: dict[str, Any] = {}

    while step < cfg.total_timesteps:
        t0 = time.time()
        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        for _ in range(cfg.batch_size):
            episode = {"obs": [], "actions": [], "log_prob": [], "reward": [], "states": [], "done": [], "avail_actions": []}
            obs, _ = env.reset(seed=cfg.seed + step)
            ep_reward, ep_length = 0.0, 0
            done, truncated = False, False
            while not done and not truncated:
                avail_action = env.get_avail_actions()
                state = env.get_state()
                with torch.no_grad():
                    actions, log_probs = actor.act(
                        torch.from_numpy(obs).float().to(device),
                        avail_action=torch.from_numpy(avail_action).bool().to(device),
                    )
                next_obs, reward, done, truncated, _infos = env.step(actions.cpu().numpy())
                ep_reward += float(reward)
                ep_length += 1
                step += 1
                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu().numpy())
                episode["log_prob"].append(log_probs.cpu().numpy())
                episode["reward"].append(float(reward))
                episode["done"].append(bool(done))
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)
                obs = next_obs
            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        b_obs, b_actions, b_log_probs, b_reward, b_states, b_avail_actions, b_done, b_mask = rb.get_batch()
        return_lambda, advantages = compute_lambda_returns(cfg, critic, b_actions, b_reward, b_states, b_mask, device)
        if cfg.normalize_advantage:
            adv_std = advantages.mean(dim=-1)[b_mask].std()
            advantages = (advantages - advantages.mean(dim=-1)[b_mask].mean()) / (adv_std + 1e-6)
        if cfg.normalize_return:
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - return_lambda.mean(dim=-1)[b_mask].mean()) / (ret_std + 1e-6)

        train_metrics = update_mappo(cfg, env, actor, critic, actor_optimizer, critic_optimizer, b_obs, b_actions, b_log_probs, return_lambda, advantages, b_states, b_avail_actions, b_mask)
        update += 1
        metrics = {
            "global_step": int(step),
            "update": int(update),
            "rollout_reward_mean": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "rollout_reward_std": float(np.std(ep_rewards)) if ep_rewards else 0.0,
            "rollout_episode_length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "seconds": float(time.time() - t0),
            **train_metrics,
        }
        if update % cfg.eval_steps == 0 or step >= cfg.total_timesteps:
            eval_metrics = evaluate_policy(cfg, actor, eval_env=eval_env, episodes=cfg.num_eval_ep, gif_path=None)
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            best_eval = max(best_eval, float(eval_metrics["mean_reward"]))
        metrics["best_eval_reward"] = float(best_eval)
        history.append(metrics)
        log_metrics(writer, metrics, step)
        append_jsonl(run_dir / "training_stats.jsonl", metrics)
        print_update(cfg, metrics)
        save_checkpoint(run_dir / "last.pt", cfg, actor, critic, actor_optimizer, critic_optimizer, env, step, update, best_eval, history)
        if metrics.get("eval_mean_reward", -float("inf")) >= best_eval:
            save_checkpoint(run_dir / "best.pt", cfg, actor, critic, actor_optimizer, critic_optimizer, env, step, update, best_eval, history)

    writer.close()
    env.close()
    eval_env.close()
    return metrics


def compute_lambda_returns(cfg, critic, b_actions, b_reward, b_states, b_mask, device):
    return_lambda = torch.zeros_like(b_actions).float().to(device)
    advantages = torch.zeros_like(b_actions).float().to(device)
    with torch.no_grad():
        for ep_idx in range(return_lambda.size(0)):
            ep_len = b_mask[ep_idx].sum()
            last_return_lambda = 0
            for t in reversed(range(ep_len)):
                if t == (ep_len - 1):
                    next_value = 0
                else:
                    next_value = critic(x=b_states[ep_idx, t + 1])
                return_lambda[ep_idx, t] = last_return_lambda = b_reward[ep_idx, t] + cfg.gamma * (
                    cfg.td_lambda * last_return_lambda + (1 - cfg.td_lambda) * next_value
                )
                advantages[ep_idx, t] = return_lambda[ep_idx, t] - critic(x=b_states[ep_idx, t])
    return return_lambda, advantages


def update_mappo(cfg, env, actor, critic, actor_optimizer, critic_optimizer, b_obs, b_actions, b_log_probs, return_lambda, advantages, b_states, b_avail_actions, b_mask):
    actor_losses = []
    critic_losses = []
    entropies_bonuses = []
    kl_divergences = []
    clipped_ratios = []
    for _ in range(cfg.epochs):
        actor_loss = 0
        critic_loss = 0
        entropies = 0
        kl_divergence = 0
        clipped_ratio = 0
        for t in range(b_obs.size(1)):
            current_logits = actor.logits(x=b_obs[:, t], avail_action=b_avail_actions[:, t])
            current_dist = Categorical(logits=current_logits)
            current_logprob = current_dist.log_prob(b_actions[:, t])
            log_ratio = current_logprob - b_log_probs[:, t]
            ratio = torch.exp(log_ratio)
            pg_loss1 = advantages[:, t] * ratio
            pg_loss2 = advantages[:, t] * torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)
            pg_loss = torch.min(pg_loss1[b_mask[:, t]], pg_loss2[b_mask[:, t]]).mean(dim=-1).sum()
            entropy_loss = current_dist.entropy()[b_mask[:, t]].mean(dim=-1).sum()
            entropies += entropy_loss
            actor_loss += -pg_loss - cfg.entropy_coef * entropy_loss
            current_values = critic(x=b_states[:, t]).expand(-1, env.n_agents)
            value_loss = F.mse_loss(current_values[b_mask[:, t]], return_lambda[:, t][b_mask[:, t]]) * (b_mask[:, t].sum())
            critic_loss += value_loss
            kl_divergence += (((ratio - 1) - log_ratio)[b_mask[:, t]].mean(dim=-1).sum())
            clipped_ratio += (((ratio - 1.0).abs() > cfg.ppo_clip)[b_mask[:, t]].float().mean(dim=-1).sum())
        actor_loss /= b_mask.sum()
        critic_loss /= b_mask.sum()
        entropies /= b_mask.sum()
        kl_divergence /= b_mask.sum()
        clipped_ratio /= b_mask.sum()
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
        critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
        if cfg.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=cfg.clip_gradients)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=cfg.clip_gradients)
        actor_optimizer.step()
        critic_optimizer.step()
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropies_bonuses.append(entropies.item())
        kl_divergences.append(kl_divergence.item())
        clipped_ratios.append(float(clipped_ratio.cpu()))
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "entropy": float(np.mean(entropies_bonuses)),
        "kl_divergence": float(np.mean(kl_divergences)),
        "clipped_ratio": float(np.mean(clipped_ratios)),
        "actor_gradient": float(actor_gradient),
        "critic_gradient": float(critic_gradient),
    }


def evaluate_policy(cfg: CleanMAPPOConfig, actor: Actor, eval_env=None, episodes: int | None = None, gif_path: str | None = None) -> dict[str, float]:
    own_env = eval_env is None
    env = eval_env or make_env(cfg, "eval", render_mode="rgb_array" if gif_path else None)
    device = next(actor.parameters()).device
    rewards, lengths, frames_by_episode = [], [], []
    try:
        for ep in range(episodes or cfg.num_eval_ep):
            obs, _ = env.reset(seed=cfg.seed + 1000 + ep)
            done = truncated = False
            ep_reward, ep_len = 0.0, 0
            frames = []
            while not done and not truncated:
                with torch.no_grad():
                    logits = actor.logits(
                        torch.from_numpy(obs).float().to(device),
                        avail_action=torch.from_numpy(env.get_avail_actions()).bool().to(device),
                    )
                    actions = torch.argmax(logits, dim=-1).cpu().numpy()
                obs, reward, done, truncated, _infos = env.step(actions)
                ep_reward += float(reward)
                ep_len += 1
                if gif_path:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            rewards.append(ep_reward)
            lengths.append(ep_len)
            if frames:
                frames_by_episode.append(frames)
    finally:
        if own_env:
            env.close()
    if gif_path and frames_by_episode:
        write_gif_mosaic(gif_path, frames_by_episode)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "var_reward": float(np.var(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def save_checkpoint(path, cfg, actor, critic, actor_optimizer, critic_optimizer, env, step, update, best_eval, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "obs_size": env.get_obs_size(),
            "state_size": env.get_state_size(),
            "action_size": env.get_action_size(),
            "n_agents": env.n_agents,
            "global_step": step,
            "update": update,
            "best_eval_reward": best_eval,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device)
    cfg = CleanMAPPOConfig.from_dict(payload["config"])
    actor = Actor(payload["obs_size"], cfg.actor_hidden_dim, cfg.actor_num_layers, payload["action_size"]).to(device)
    critic = Critic(payload["state_size"], cfg.critic_hidden_dim, cfg.critic_num_layers).to(device)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return cfg, actor, critic, payload


def write_gif_mosaic(path: str | Path, episodes: list[list[Any]]) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(frames) for frames in episodes)
    first = np.asarray(episodes[0][0])
    h, w = first.shape[:2]
    c = first.shape[2] if first.ndim == 3 else 1
    cols = int(math.ceil(math.sqrt(len(episodes))))
    rows = int(math.ceil(len(episodes) / cols))
    pad = 8
    out = []
    for t in range(max_len):
        canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, c), dtype=np.uint8)
        for idx, frames in enumerate(episodes):
            frame = np.asarray(frames[min(t, len(frames) - 1)], dtype=np.uint8)
            r, col = divmod(idx, cols)
            y, x = r * (h + pad), col * (w + pad)
            canvas[y:y + h, x:x + w, : frame.shape[2]] = frame
        out.append(canvas)
    imageio.mimsave(path, out, duration=0.08, loop=0)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def log_metrics(writer, metrics, step):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key.replace("_", "/"), value, step)


def print_update(cfg, metrics):
    print(
        "[CleanMARL MAPPO] "
        f"update={metrics.get('update')} "
        f"steps={metrics.get('global_step')}/{cfg.total_timesteps} "
        f"eval_return_mean={metrics.get('eval_mean_reward', float('nan')):.4f} "
        f"eval_return_var={metrics.get('eval_var_reward', float('nan')):.4f} "
        f"rollout_return_mean={metrics.get('rollout_reward_mean', 0.0):.4f} "
        f"actor_loss={metrics.get('actor_loss', 0.0):.4f} "
        f"critic_loss={metrics.get('critic_loss', 0.0):.4f} "
        f"entropy={metrics.get('entropy', 0.0):.4f}",
        flush=True,
    )
