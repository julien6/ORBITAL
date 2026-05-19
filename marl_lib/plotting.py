from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_training_figures(history: list[dict[str, Any]], figures_dir: str | Path) -> None:
    if not history:
        return
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    steps = [float(row.get("global_step", idx)) for idx, row in enumerate(history)]

    def values(key: str) -> list[float]:
        out = []
        for row in history:
            value = row.get(key)
            out.append(float(value) if isinstance(value, (int, float)) else float("nan"))
        return out

    def save_line(path: str, series: list[tuple[str, list[float]]], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        for label, vals in series:
            ax.plot(steps, vals, label=label, linewidth=1.8)
        ax.set_xlabel("global environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / path)
        plt.close(fig)

    save_line(
        "learning_curve.png",
        [
            ("eval mean return", values("eval_mean_reward")),
            ("rollout mean return", values("rollout_reward_mean")),
        ],
        "episode return",
    )

    eval_mean = values("eval_mean_reward")
    eval_std = values("eval_std_reward")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(steps, eval_mean, label="eval mean return", linewidth=1.8)
    lower = [m - s for m, s in zip(eval_mean, eval_std)]
    upper = [m + s for m, s in zip(eval_mean, eval_std)]
    ax.fill_between(steps, lower, upper, alpha=0.2, label="+/- 1 std")
    ax.set_xlabel("global environment steps")
    ax.set_ylabel("episode return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "eval_return_with_variance.png")
    plt.close(fig)

    save_line(
        "losses.png",
        [
            ("policy loss", values("policy_loss")),
            ("value loss", values("value_loss")),
        ],
        "loss",
    )
    save_line(
        "policy_diagnostics.png",
        [
            ("entropy", values("entropy")),
            ("approx KL", values("approx_kl")),
            ("clip fraction", values("clip_fraction")),
        ],
        "diagnostic value",
    )
    save_line(
        "delivery_curve.png",
        [
            ("eval mean delivered", values("eval_mean_delivered")),
        ],
        "delivered data",
    )
