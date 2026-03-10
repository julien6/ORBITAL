from __future__ import annotations

import math

import numpy as np

from orbital import parallel_env3d


ACTION_OBSERVE = 0
ACTION_RELAY = 1
ACTION_ORBIT_DOWN = 2
ACTION_ORBIT_UP = 3
ACTION_LOWPOWER = 4
ACTION_IDLE = 6


def scripted_actions(env, step_idx: int) -> dict[str, int]:
    """Simple deterministic policy to keep the scene visually rich."""
    actions: dict[str, int] = {}
    for i, agent in enumerate(env.agents):
        if i == 0:
            # Leader stays near ground contact and relays often.
            actions[agent] = ACTION_RELAY
        elif i == len(env.agents) - 1:
            # Tail agent drifts and will be forced isolated for blocked downlink.
            actions[agent] = ACTION_ORBIT_UP if (step_idx % 2 == 0) else ACTION_ORBIT_DOWN
        elif i % 3 == 0:
            actions[agent] = ACTION_OBSERVE
        elif i % 3 == 1:
            actions[agent] = ACTION_RELAY
        else:
            actions[agent] = ACTION_LOWPOWER if (step_idx % 4 == 0) else ACTION_IDLE
    return actions


def force_overview_state(env) -> None:
    """Inject small deterministic tweaks so all key overlays appear together."""
    core = env.core
    n = core.num_agents
    if n < 2:
        return

    # Keep some buffered data so downlink overlays remain visible.
    core.buffered_data = np.maximum(core.buffered_data, 1.5)

    # Force one agent in direct contact with first ground station.
    core.orbit_theta[0] = float(core.ground_thetas[0])
    core.orbit_phi[0] = float(core.ground_phis[0])

    # Force last agent roughly opposite to ground station and disconnected.
    core.orbit_theta[n - 1] = float((core.ground_thetas[0] + math.pi) % (2.0 * math.pi))
    core.orbit_phi[n - 1] = float(-core.ground_phis[0])

    core._refresh_cartesian_positions()
    core.update_comm_graph()

    # Keep links globally but isolate one satellite for "blocked downlink" + "isolated" view.
    core.comm_adj[n - 1, :] = False
    core.comm_adj[:, n - 1] = False

    # Keep one compromised satellite to show alert overlay.
    if n > 2:
        core.compromised_for[2] = max(core.compromised_for[2], 3)


def main() -> None:
    env = parallel_env3d(
        num_satellites=10,
        max_steps=240,
        energy_budget=220.0,
        recharge_rate=1.8,
        comm_radius=3,
        p_link_drop=0.02,
        adversarial_rate=0.0,
        compromise_duration=10,
        ground_station_thetas=(
            -math.pi / 2.0,
            0.0,
            math.pi / 2.0,
            math.pi,
        ),
        ground_station_phis=(
            0.0,
            0.25,
            -0.25,
            0.0,
        ),
        ground_contact_angle=0.42,
        num_tasks=12,
        num_debris_clouds=6,
        render_projection="3d",
        render_quality="ultra_low",
        render_mode="human",
    )

    obs, infos = env.reset(seed=7)
    for step_idx in range(env.config.max_steps):
        if not env.agents:
            break
        actions = scripted_actions(env, step_idx)
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Force a global "feature parity" view before rendering.
        force_overview_state(env)
        env.render()

    env.close()


if __name__ == "__main__":
    main()

