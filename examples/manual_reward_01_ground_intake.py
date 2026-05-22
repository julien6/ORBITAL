from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbital import parallel_env


REL_GRN = 1
IDLE = 7
GROUND_CONTACT = 6


def manual_joint_policy(obs: dict[str, object]) -> dict[str, int]:
    """Use direct ground windows only to fetch the task catalog."""
    return {
        agent: REL_GRN if agent_obs[GROUND_CONTACT] > 0.5 else IDLE for agent, agent_obs in obs.items()
    }


def print_step_reward(step: int, rewards: dict[str, float], infos: dict[str, dict]) -> float:
    team_reward = next(iter(rewards.values()), 0.0)
    components = next(iter(infos.values()))["reward_components"]
    print(
        f"step={step:03d} team_reward={team_reward:>8.3f} "
        f"intake={components['ground_task_intake']:>5.2f} "
        f"knowledge={components['knowledge']:>5.2f} "
        f"task={components['task']:>5.2f} "
        f"delivery={components['delivery']:>5.2f}"
    )
    return team_reward


def main() -> None:
    env = parallel_env(
        render_mode="human",
        task_knowledge_mode="local_discovery",
        enable_local_task_discovery=True,
    )
    obs, infos = env.reset(seed=42)
    cumulative_reward = 0.0
    for step in range(env.config.max_steps):
        actions = manual_joint_policy(obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        cumulative_reward += print_step_reward(step, rewards, infos)
        env.render()
        if not env.agents:
            break
    print(f"final_cumulative_team_reward={cumulative_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
