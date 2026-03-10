# ORBITAL

<p align="center">
  <img src="assets/logo.svg" width="400">
</p>

ORBITAL (*Orbital Resilient Benchmark for Interactive Task-aware Autonomous Learning*) is a PettingZoo multi-agent environment where a satellite constellation must keep a mission running under resource pressure and cyber disruption.

It is designed as a neutral benchmark: the environment does not impose a coordination method. You can plug in value-based MARL, policy gradient, planning, centralized training, decentralized execution, or custom protocols.

The collective objective is simple to state and hard to optimize: maximize useful data delivered to ground while keeping the fleet alive, connected, and resilient.

## Why ORBITAL

Many multi-agent benchmarks focus on either coordination, communication, or adversarial robustness. ORBITAL combines all three in one loop:

* constrained agents with finite energy
* dynamic communication graph with link drops
* non-stationary tasks that appear, expire, and change priority
* data relay to a fixed ground station
* stochastic cyber compromise events affecting observations/actions/comms
* orbital debris fields inducing conjunction-risk pressure

The goal is not just to complete tasks, but to do it reliably when the system is stressed.

## Scenario Overview

Each episode simulates `N` satellites on orbital trajectories around Earth.
Default mode is 2D ( `theta` , radius), with an optional 3D mode ( `theta` , radius, `phi` ).

At each step:

1. satellites choose actions (`Observe`, `Relay`, `OrbitDown`, `OrbitUp`, `LowPower`, `CyberScan`, `Idle`)
2. energy is consumed based on action costs
3. tasks may be serviced and turned into buffered data
4. buffered data can be delivered when relay conditions are met
5. communication links are recomputed from distance + random link drops
6. adversarial events may compromise satellites
7. debris clouds evolve and induce local collision-risk pressure
8. rewards are computed from mission and resilience outcomes

Episodes end by horizon ( `max_steps` ) or mission collapse (for example too few alive satellites).

## Operational Intuition (Mission Story)

Think of ORBITAL as a repeated loop with two distinct phases:

1. `Observe` turns local opportunities into buffered data on a satellite.
2. `Relay` converts buffered data into mission value only when communication paths to a ground station exist.

This deliberate split is where most difficulty comes from: a fleet can look productive locally while still failing globally if data never reaches ground.

Ideal policy profile:

* some satellites prioritize acquisition when local task density is high
* some satellites maintain delivery continuity and network utility
* some satellites absorb stress periods (energy/cyber) with safer actions
* behavior switches over time as topology, tasks, and threats evolve

Naive policy profile:

* all satellites overuse `Observe`
* buffers saturate
* relay opportunities are missed
* energy and connectivity degrade, then mission value collapses

Concrete episode pattern (typical failure mode):

* at one step, several satellites become temporarily isolated by link drops + line-of-sight constraints
* a myopic policy keeps observing despite no delivery path
* a strategic policy reallocates actions toward connectivity and relay, then resumes acquisition

## Core Mechanics

### 1) Mission and Tasks

* Tasks are spatial targets in orbital coordinates `(theta, radius)`.
* `Observe` services nearby tasks in local orbital vicinity.
* Servicing generates buffered data on the satellite.
* Data only contributes fully when it is relayed through communication paths toward the ground station.

This separates sensing from delivery, which creates realistic tradeoffs between coverage and communication.

### 2) Communication

* Satellites form a graph each step:
  + edge exists if orbital Euclidean distance is within communication radius
  + then edge may drop with probability `p_link_drop`
* Relay can succeed on direct ground-contact windows or through available network paths.

This induces non-stationarity in who can collaborate at each step.

### 3) Energy and Survival

* Every action consumes energy (`energy_costs`).
* Optional sunlight recharge can restore energy (`enable_recharge`,          `recharge_rate`).
* Satellites at zero energy become effectively inactive.

Energy management is central: short-term gains can cause long-term mission collapse.

### 4) Cyber Adversary

* With probability `adversarial_rate`, a healthy satellite can be compromised for `compromise_duration` steps.
* Effects depend on `spoof_mode`:
  + `obs_spoof`: tampered observations
  + `action_noise`: occasional action perturbation
  + `comm_jam`: relay interference
* `CyberScan` can reduce compromise impact/duration.

This creates deceptive local information and forces policies to reason under uncertainty.

### 5) Orbital Debris and Conjunction Risk

* Optional drifting debris clouds are sampled in orbital coordinates.
* Each satellite gets a local conjunction-risk proxy (`Pc` estimate) from nearby debris density.
* `OrbitDown` / `OrbitUp` act as simple avoidance maneuvers that reduce immediate risk.
* If risk remains high without mitigation, the environment applies risk penalties and rare collision events.

This models risk-aware autonomy (screening + mitigation) rather than static obstacle avoidance.

## Observation and Action Spaces

ORBITAL intentionally uses **fixed-size vector observations only** for MARL compatibility and reproducibility.

* Observation: `Box(shape=(16,), dtype=np.float32)`
* Action: `Discrete(7)`

Action IDs:

| ID | Action |
|---:|---|
| 0 | Observe |
| 1 | Relay |
| 2 | OrbitDown |
| 3 | OrbitUp |
| 4 | LowPower |
| 5 | CyberScan |
| 6 | Idle |

Full observation semantics are documented in `docs/ENV_SPECS.md` .

## Reward Design

Default reward mode is shared team reward ( `reward_mode="shared"` ), with configurable weights:

* `+ task`: serviced task value
* `+ delivery`: data delivered to ground
* `- energy`: action energy consumption
* `- isolation`: disconnected alive satellites
* `- failure`: satellite losses
* `- cyber`: adversarial impact
* `- debris_risk`: conjunction-risk exposure
* `- collision`: collision events

`reward_mode="local"` is also available.

## Installation

```bash
pip install -e .
pip install -e '.[render]'     # pygame rendering
pip install -e '.[render3d]'   # pyvista/vtk 3D rendering
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
        action = 6  # Idle
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

3D entrypoints are also available:

```python
from orbital import env3d, parallel_env3d
```

## Rendering

Supported render modes:

* `render_mode="human"`: live window
* `render_mode="rgb_array"`: returns `np.uint8` frame arrays

Projection modes:

* `render_projection="2d"`: pygame orbital map view
* `render_projection="3d"`: pyvista true 3D view (Earth/satellites/tasks/debris as spheres)
* `render_quality="ultra_low|low|medium|high"`: controls 3D rendering load/smoothness (default: `medium`)

Both projections preserve the same semantic colors for mission entities.

Example:

```bash
python examples/human_render.py
```

## Key Configuration Parameters

Common knobs:

* `num_satellites`,          `grid_size`
* `num_tasks`,          `task_spawn_rate`,          `task_priority_mode`
* `energy_budget`, `energy_costs`, `enable_recharge`,          `recharge_rate`
* `comm_radius`,  `p_link_drop`
* `orbit_min_radius`,         `orbit_max_radius`,         `kepler_constant`,         `orbit_shift_step`,         `earth_radius`
* `ground_station_thetas`,         `ground_contact_angle`
* `ground_station_phis`,        `world_dim`,        `inclination_max`,        `render_projection`
* `render_quality`
* `adversarial_rate`,          `compromise_duration`,  `spoof_mode`
* `enable_debris`,         `num_debris_clouds`,         `debris_spawn_rate`,         `debris_decay`,         `debris_drift_std`
* `debris_spread_min`,         `debris_spread_max`,         `debris_risk_gain`,         `pc_alert_threshold`,         `pc_collision_scale`
* `debris_mitigation_factor`
* `reward_weights`,          `reward_mode`
* `max_steps`,          `render_mode`

For exact defaults, see `orbital/envs/core/config.py` .

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

* robust MARL under non-stationarity
* communication-aware policy learning
* cyber-resilience studies in cooperative settings
* controlled ablations on energy, topology, and attack intensity

It is not a high-fidelity orbital simulator; it is a research benchmark with intentionally simplified but interacting constraints.

## Why Not Only DCOP / Rules / Vanilla MARL?

ORBITAL is compatible with many controllers, but each family has limits in this setting:

* fixed rules are easy to audit but brittle under non-stationary tasks and cyber perturbations
* classical DCOP-style formulations are strong for explicit coordination but are harder to keep adaptive under partial observability and stochastic dynamics
* vanilla MARL adapts well but does not provide explicit role/mission control by default

This is why ORBITAL is suited to test hybrid approaches: keep learning-based adaptation while injecting explicit operational structure.
