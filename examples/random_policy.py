from orbital import parallel_env


def main():
    env = parallel_env(num_satellites=6)
    obs, infos = env.reset(seed=1)
    for _ in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if not env.agents:
            break
    env.close()


if __name__ == "__main__":
    main()
