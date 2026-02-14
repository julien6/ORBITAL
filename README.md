# ORBITAL

![ORBITAL logo](assets/logo.svg)

ORBITAL (*Orbital Resilient Benchmark for Interactive Task-aware Autonomous Learning*) is a PettingZoo multi-agent environment where a satellite constellation must keep a mission running under resource pressure and cyber disruption.

It is designed as a neutral benchmark: the environment does not impose a coordination method. You can plug in value-based MARL, policy gradient, planning, centralized training, decentralized execution, or custom protocols.

## Why ORBITAL

Many multi-agent benchmarks focus on either coordination, communication, or adversarial robustness. ORBITAL combines all three in one loop:

- constrained agents with finite energy
- dynamic communication graph with link drops
- non-stationary tasks that appear, expire, and change priority
- data relay to a fixed ground station
- stochastic cyber compromise events affecting observations/actions/comms

The goal is not just to complete tasks, but to do it reliably when the system is stressed.

## Scenario Overview

Each episode simulates `N` satellites on a 2D grid.

At each step:

1. satellites choose actions (`Observe`, `Relay`, `Move`, `LowPower`, `CyberScan`, `Idle`)
2. energy is consumed based on action costs
3. tasks may be serviced and turned into buffered data
4. buffered data can be delivered when relay conditions are met
5. communication links are recomputed from distance + random link drops
6. adversarial events may compromise satellites
7. rewards are computed from mission and resilience outcomes

Episodes end by horizon (`max_steps`) or mission collapse (for example too few alive satellites).

## Core Mechanics (Pedagogical View)

### 1) Mission and Tasks

- Tasks are spatial targets on the grid.
- `Observe` services nearby tasks (Manhattan distance `<= 1`).
- Servicing generates buffered data on the satellite.
- Data only contributes fully when it is relayed toward the ground station at `(0, 0)`.

This separates sensing from delivery, which creates realistic tradeoffs between coverage and communication.

### 2) Communication

- Satellites form a graph each step:
  - edge exists if Manhattan distance `<= comm_radius`
  - then edge may drop with probability `p_link_drop`
- Relay can succeed near ground or through available network paths.

This induces non-stationarity in who can collaborate at each step.

### 3) Energy and Survival

- Every action consumes energy (`energy_costs`).
- Optional sunlight recharge can restore energy (`enable_recharge`, `recharge_rate`).
- Satellites at zero energy become effectively inactive.

Energy management is central: short-term gains can cause long-term mission collapse.

### 4) Cyber Adversary

- With probability `adversarial_rate`, a healthy satellite can be compromised for `compromise_duration` steps.
- Effects depend on `spoof_mode`:
  - `obs_spoof`: tampered observations
  - `action_noise`: occasional action perturbation
  - `comm_jam`: relay interference
- `CyberScan` can reduce compromise impact/duration.

This creates deceptive local information and forces policies to reason under uncertainty.

## Observation and Action Spaces

ORBITAL intentionally uses **fixed-size vector observations only** for MARL compatibility and reproducibility.

- Observation: `Box(shape=(14,), dtype=np.float32)`
- Action: `Discrete(6)`

Action IDs:

| ID | Action |
|---:|---|
| 0 | Observe |
| 1 | Relay |
| 2 | Move |
| 3 | LowPower |
| 4 | CyberScan |
| 5 | Idle |

Full observation semantics are documented in `docs/ENV_SPECS.md`.

## Reward Design

Default reward mode is shared team reward (`reward_mode="shared"`), with configurable weights:

- `+ task`: serviced task value
- `+ delivery`: data delivered to ground
- `- energy`: action energy consumption
- `- isolation`: disconnected alive satellites
- `- failure`: satellite losses
- `- cyber`: adversarial impact

`reward_mode="local"` is also available.

## Installation

```bash
pip install -e .
pip install -e '.[render]'     # pygame rendering
pip install -e '.[dev]'        # tests/dev tooling
```

## Quickstart (AEC API)

```python
from orbital import env

e = env(num_satellites=8, max_steps=128, render_mode=None)
e.reset(seed=7)

for agent in e.agent_iter():
    obs, reward, terminated, truncated, info = e.last()
    if terminated or truncated:
        action = 5  # Idle
    else:
        action = e.action_space(agent).sample()
    e.step(action)

e.close()
```

## Quickstart (Parallel API)

```python
from orbital import parallel_env

e = parallel_env(num_satellites=8, max_steps=128, render_mode=None)
obs, infos = e.reset(seed=7)

while e.agents:
    actions = {agent: e.action_space(agent).sample() for agent in e.agents}
    obs, rewards, terminations, truncations, infos = e.step(actions)

e.close()
```

## Rendering

Supported render modes:

- `render_mode="human"`: live pygame window
- `render_mode="rgb_array"`: returns `np.uint8` frame arrays

Human rendering includes satellites, IDs, links, tasks, ground station, energy bars, compromise highlighting, and mission HUD.

Example:

```bash
python examples/human_render.py
```

## Key Configuration Parameters

Common knobs:

- `num_satellites`, `grid_size`
- `num_tasks`, `task_spawn_rate`, `task_priority_mode`
- `energy_budget`, `energy_costs`, `enable_recharge`, `recharge_rate`
- `comm_radius`, `p_link_drop`
- `adversarial_rate`, `compromise_duration`, `spoof_mode`
- `reward_weights`, `reward_mode`
- `max_steps`, `render_mode`

For exact defaults, see `orbital/envs/core/config.py`.

## Project Layout

```text
orbital/
  envs/
    orbital_aec.py
    orbital_parallel.py
    core/
      config.py
      dynamics.py
      reward.py
      spaces.py
    rendering/
      pygame_renderer.py
examples/
  random_policy.py
  human_render.py
tests/
docs/
  ENV_SPECS.md
```

## Testing

```bash
pytest -q
```

The test suite includes PettingZoo API conformance and deterministic-seed checks.

## Intended Use

ORBITAL is useful for:

- robust MARL under non-stationarity
- communication-aware policy learning
- cyber-resilience studies in cooperative settings
- controlled ablations on energy, topology, and attack intensity

It is not a high-fidelity orbital simulator; it is a research benchmark with intentionally simplified but interacting constraints.
