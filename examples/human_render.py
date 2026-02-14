from orbital import parallel_env


def main():
    env = parallel_env(num_satellites=8, render_mode="human")
    obs, infos = env.reset(seed=42)
    for _ in range(200):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        if not env.agents:
            break
    env.close()


if __name__ == "__main__":
    main()
