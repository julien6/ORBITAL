from orbital import parallel_env, parallel_env3d


def rollout(seed=7, steps=40, world_dim=2):
    ctor = parallel_env3d if world_dim == 3 else parallel_env
    e = ctor(num_satellites=5, max_steps=steps)
    obs, infos = e.reset(seed=seed)
    n_actions = e.action_space(e.possible_agents[0]).n
    trace = []
    for t in range(steps):
        if not e.agents:
            break
        actions = {a: (t + i) % n_actions for i, a in enumerate(e.agents)}
        obs, rewards, terms, truncs, infos = e.step(actions)
        snap = (
            tuple(sorted((k, round(v, 6)) for k, v in rewards.items())),
            tuple(sorted((k, infos[k]["energy"]) for k in infos)),
            tuple(sorted((k, infos[k]["compromised"]) for k in infos)),
        )
        trace.append(snap)
    e.close()
    return trace


def test_deterministic_rollout():
    assert rollout(seed=11) == rollout(seed=11)


def test_deterministic_rollout_3d():
    assert rollout(seed=11, world_dim=3) == rollout(seed=11, world_dim=3)
