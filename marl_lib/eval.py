from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import numpy as np
import torch
from torch.distributions import Categorical

from marl_lib.config import MAPPOConfig
from marl_lib.envs import make_env
from marl_lib.preprocessing import preprocess_observation


class EvalRunner:
    def __init__(self, config: MAPPOConfig, model: torch.nn.Module, agent_order: list[str]):
        self.config = config
        self.model = model
        self.agent_order = list(agent_order)
        self.device = next(model.parameters()).device

    def run(self, episodes: int | None = None, gif_path: str | None = None, deterministic: bool = True) -> dict[str, float]:
        episodes = self.config.eval_episodes if episodes is None else episodes
        env_kwargs = dict(self.config.env_kwargs)
        if gif_path is not None:
            env_kwargs["render_mode"] = "rgb_array"
        env = make_env(self.config.env_id, env_kwargs,
                       self.config.organization_path, mma_mode="eval", seed=self.config.seed)
        rewards = []
        delivered = []
        lengths = []
        episode_frames: list[list[Any]] = []
        try:
            for ep in range(episodes):
                obs, infos = env.reset(seed=self.config.seed + 1000 + ep)
                ep_reward = 0.0
                step_count = 0
                frames = []
                while getattr(env, "agents", []):
                    actions = {}
                    for agent in list(env.agents):
                        obs_t = torch.as_tensor(
                            self._preprocess_obs(obs[agent]), dtype=torch.float32, device=self.device).unsqueeze(0)
                        logits = self.model.action_logits(obs_t)
                        if deterministic:
                            action = torch.argmax(logits, dim=-1)
                        else:
                            action = Categorical(logits=logits).sample()
                        actions[agent] = int(action.item())
                    obs, step_rewards, terms, truncs, infos = env.step(actions)
                    step_count += 1
                    ep_reward += float(np.mean(list(step_rewards.values()))
                                       ) if step_rewards else 0.0
                    if gif_path is not None:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    if any(terms.values()) or any(truncs.values()):
                        break
                rewards.append(ep_reward)
                lengths.append(step_count)
                if frames:
                    episode_frames.append(frames)
                if infos:
                    first_info = next(iter(infos.values()))
                    delivered.append(float(first_info.get(
                        "episode", {}).get("delivered", 0.0)))
        finally:
            env.close()
        if gif_path is not None and episode_frames:
            self._write_gif(gif_path, self._mosaic_frames(episode_frames))
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "var_reward": float(np.var(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_delivered": float(np.mean(delivered)) if delivered else 0.0,
        }

    def _write_gif(self, path: str, frames: list[Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.mimsave(path, frames, duration=0.08, loop=0)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install imageio or `pip install -e '.[train]'` to export GIFs") from exc

    def _mosaic_frames(self, episodes: list[list[Any]]) -> list[np.ndarray]:
        if len(episodes) == 1:
            return [np.asarray(frame) for frame in episodes[0]]
        max_len = max(len(frames) for frames in episodes)
        first = np.asarray(episodes[0][0])
        height, width = int(first.shape[0]), int(first.shape[1])
        channels = int(first.shape[2]) if first.ndim == 3 else 1
        cols = int(math.ceil(math.sqrt(len(episodes))))
        rows = int(math.ceil(len(episodes) / cols))
        pad = 8
        canvas_shape = (
            rows * height + (rows - 1) * pad,
            cols * width + (cols - 1) * pad,
            channels,
        )
        mosaic = []
        for t in range(max_len):
            canvas = np.zeros(canvas_shape, dtype=np.uint8)
            for idx, frames in enumerate(episodes):
                frame = np.asarray(
                    frames[min(t, len(frames) - 1)], dtype=np.uint8)
                if frame.shape[:2] != (height, width):
                    frame = frame[:height, :width]
                r = idx // cols
                c = idx % cols
                y = r * (height + pad)
                x = c * (width + pad)
                canvas[y:y + height, x:x + width, : frame.shape[2]] = frame
            mosaic.append(canvas)
        return mosaic

    def _preprocess_obs(self, obs) -> np.ndarray:
        return preprocess_observation(
            obs,
            image_downsample=self.config.image_downsample,
            image_grayscale=self.config.image_grayscale,
        )
