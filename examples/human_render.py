from orbital import parallel_env, parallel_env3d


def main():
    # Demo-oriented settings: keep satellites alive long enough for visual inspection
    # with random actions (not intended as benchmark defaults).
    use_3d = False
    use_3d_projection = False
    ctor = parallel_env3d if use_3d else parallel_env
    env = ctor(
        num_satellites=8,
        max_steps=200,
        energy_budget=180.0,
        recharge_rate=1.4,
        p_link_drop=0.02,
        adversarial_rate=0.01,
        render_projection="3d" if use_3d_projection else "2d",
        render_quality="ultra_low",
        render_mode="human",
    )
    obs, infos = env.reset(seed=42)
    # Keep one render per environment step to avoid multiplying render latency.
    frames_per_decision = 1
    for _ in range(env.config.max_steps):
        actions = {agent: env.action_space(
            agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for _ in range(frames_per_decision):
            env.render()
        if not env.agents:
            break
    env.close()


if __name__ == "__main__":
    main()
