# ORBITAL

ORBITAL (*Orbital Resilient Benchmark for Interactive Task-aware Autonomous Learning*) is a PettingZoo multi-agent environment focused on resilient satellite coordination under communication, energy, and cyber constraints.

## Features

- PettingZoo **AEC** API (`orbital.env()`)
- PettingZoo **Parallel** API (`orbital.parallel_env()`)
- Fixed-size vector observations (size 14) for MARL compatibility
- Dynamic tasking, communication graph drops, energy/recharge, and cyber compromise events
- `render_mode="human"` and `render_mode="rgb_array"` via pygame

## Installation

```bash
pip install -e .
# optional rendering / dev extras
pip install -e '.[render,dev]'
```

## Quickstart

```python
from orbital import env

e = env(num_satellites=8, render_mode=None)
e.reset(seed=7)
for agent in e.agent_iter(max_iter=128):
    obs, reward, term, trunc, info = e.last()
    action = 5 if term or trunc else e.action_space(agent).sample()
    e.step(action)
```

## Actions

| ID | Action |
|---:|---|
| 0 | Observe |
| 1 | Relay |
| 2 | Move |
| 3 | LowPower |
| 4 | CyberScan |
| 5 | Idle |

See `docs/ENV_SPECS.md` for complete observation/reward details.
