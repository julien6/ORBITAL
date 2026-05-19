from __future__ import annotations

from pathlib import Path

import torch

from marl_lib.config import MAPPOConfig
from marl_lib.eval import EvalRunner
from marl_lib.preprocessing import preprocess_observation
from marl_lib.trainer import MAPPOTrainer


def _cfg(tmp_path: Path) -> MAPPOConfig:
    return MAPPOConfig(
        env_kwargs={"num_satellites": 2, "max_steps": 12, "enable_debris": False, "adversarial_rate": 0.0},
        run_dir=str(tmp_path),
        seed=3,
        total_steps=16,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=4,
        eval_interval=1000,
        organization_path=None,
    )


def test_mappo_collects_and_updates_without_nan(tmp_path: Path):
    trainer = MAPPOTrainer(_cfg(tmp_path))
    try:
        before = [p.detach().clone() for p in trainer.model.parameters()]
        batch, metrics = trainer.collect_rollout()
        update = trainer.update(batch)
        after = list(trainer.model.parameters())
        assert batch["obs"].shape[0] > 0
        assert all(torch.isfinite(v).all() for v in batch.values())
        assert update["entropy"] >= 0.0
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        trainer.close()


def test_mappo_checkpoint_roundtrip(tmp_path: Path):
    trainer = MAPPOTrainer(_cfg(tmp_path))
    try:
        trainer.collect_rollout()
        trainer.save_checkpoint("last.pt", {"ok": True})
        restored = MAPPOTrainer(_cfg(tmp_path))
        try:
            restored.load_checkpoint(tmp_path / "last.pt")
            assert restored.global_step == trainer.global_step
        finally:
            restored.close()
    finally:
        trainer.close()


def test_eval_mosaic_frames(tmp_path: Path):
    trainer = MAPPOTrainer(_cfg(tmp_path))
    try:
        episodes = [
            [torch.zeros((4, 5, 3), dtype=torch.uint8).numpy() for _ in range(2)],
            [torch.ones((4, 5, 3), dtype=torch.uint8).numpy() for _ in range(3)],
            [torch.full((4, 5, 3), 2, dtype=torch.uint8).numpy()],
        ]
        frames = EvalRunner(trainer.config, trainer.model, trainer.agent_order)._mosaic_frames(episodes)
        assert len(frames) == 3
        assert frames[0].shape == (16, 18, 3)
    finally:
        trainer.close()


def test_image_preprocessing_downsamples_grayscale_and_normalizes():
    image = torch.full((8, 6, 3), 255, dtype=torch.uint8).numpy()
    out = preprocess_observation(image, image_downsample=2, image_grayscale=True)
    assert out.shape == (12,)
    assert float(out.max()) == 1.0
