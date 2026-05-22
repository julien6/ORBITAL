from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbital import parallel_env


OBS = 0
REL_GRN = 1
REL_SAT = 2
PWR = 5
SCAN = 6
IDLE = 7
ENERGY = 0
SUNLIGHT = 5
GROUND_CONTACT = 6
GROUND_ROUTE = 7
LOCAL_DEGREE = 8
BUFFERED_DATA = 9
BUFFER_REMAINING = 10
KNOWN_NEARBY_TASKS = 11
COMPROMISED = 15


def manual_joint_policy(obs: dict[str, object]) -> dict[str, int]:
    """Coordinate ground relays, observers, and satellite relays."""
    actions: dict[str, int] = {agent: IDLE for agent in obs}

    for agent, agent_obs in obs.items():
        if agent_obs[COMPROMISED] > 0.5:
            actions[agent] = SCAN
            continue

        if agent_obs[ENERGY] < 0.20 and agent_obs[SUNLIGHT] > 0.5:
            actions[agent] = PWR
            continue

        if agent_obs[GROUND_CONTACT] > 0.5:
            if agent_obs[BUFFERED_DATA] > 0.05 or agent_obs[KNOWN_NEARBY_TASKS] > 0.0:
                actions[agent] = REL_GRN
            else:
                actions[agent] = IDLE
            continue

        if agent_obs[BUFFERED_DATA] > 0.05 and (
            agent_obs[GROUND_ROUTE] > 0.0 or agent_obs[LOCAL_DEGREE] > 0.0
        ):
            actions[agent] = REL_SAT
            continue

        if (
            agent_obs[BUFFERED_DATA] < 0.25
            and agent_obs[KNOWN_NEARBY_TASKS] > 0.0
            and agent_obs[BUFFER_REMAINING] > 0.3
        ):
            actions[agent] = OBS
            continue

        if agent_obs[LOCAL_DEGREE] > 0.0 and agent_obs[KNOWN_NEARBY_TASKS] > 0.0:
            actions[agent] = REL_SAT
            continue

    return actions


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
