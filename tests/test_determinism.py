from orbital import parallel_env


def rollout(seed=7, steps=40):
    e = parallel_env(num_satellites=5, max_steps=steps)
    obs, infos = e.reset(seed=seed)
    trace = []
    for t in range(steps):
        if not e.agents:
            break
        actions = {a: (t + i) % 6 for i, a in enumerate(e.agents)}
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
