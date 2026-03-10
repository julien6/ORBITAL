# ORBITAL Environment Specs

## Operational objective (intuitive)

The team objective is to maximize mission value delivered to ground while preserving fleet viability.

In practice, this means balancing four pressures at every step:

1. mission productivity (service high-value tasks),
2. delivery continuity (relay buffered data to ground),
3. survivability (avoid energy collapse),
4. resilience (mitigate communication/cyber degradation).
5. conjunction safety (limit debris exposure and collision risk).

Important modeling choice: `Observe` and `Relay` are intentionally decoupled.
Servicing a task creates buffered data, but value is only realized when data is later delivered.
This is what makes myopic policies fail in long horizons.

## Observation (fixed-size vector)

Each agent receives a `np.float32` vector of size **16**:

1. normalized self energy `[0,1]`
2. normalized orbital angle `theta/(2π)` in `[0,1]`
3. normalized orbital radius in `[0,1]`
4. cyber healthy indicator (one-hot)
5. cyber suspicious indicator (one-hot)
6. cyber compromised indicator (one-hot)
7. normalized local communication degree
8. normalized nearby active task count
9. normalized nearby task priority sum
10. normalized buffered data
11. normalized local debris density
12. local conjunction-risk proxy (`Pc` estimate in `[0,1]`)
13. sunlight phase `sin(2πt/T)`
14. sunlight phase `cos(2πt/T)` in 2D mode, or normalized orbital inclination `phi` in 3D mode
15. fraction of compromised neighbors
16. alive satellite fraction

> `obs_mode` is intentionally not exposed: ORBITAL uses fixed-size vectors only.

## Action space

`Discrete(7)`:

- `0 Observe`
- `1 Relay`
- `2 OrbitDown` (transfer to a lower orbit)
- `3 OrbitUp` (transfer to a higher orbit)
- `4 LowPower`
- `5 CyberScan`
- `6 Idle`

## Typical policy contrast

Ideal coordinated behavior:

- allocate temporary role-like specializations (acquire / relay / safety),
- switch behavior with communication windows and stress intervals,
- avoid prolonged `Observe` runs when no delivery path is available.

Common failure behavior:

- over-focus on local sensing,
- under-use relay opportunities,
- accumulate undelivered buffers until energy/connectivity collapse.

## Reward

By default, `reward_mode="shared"` with weighted components:

- `+ task serviced`
- `+ data delivered to ground`
- `- energy consumed`
- `- isolated satellites`
- `- satellite failures`
- `- cyber impact`
- `- debris risk exposure`
- `- collision events`

Configure via `reward_weights` in constructor.

The default reward is intentionally non-myopic: it rewards mission output but penalizes patterns that make the fleet brittle over time (isolation, failures, cyber impact, avoidable energy drain).

## Termination / truncation

- truncation when `t >= max_steps`
- termination when mission collapse occurs (all dead or critically few alive)

## Infos

`infos[agent]` includes:

- `energy`
- `compromised`
- `local_degree`
- `buffered_data`
- `theta`
- `phi`
- `radius`
- `local_debris_density`
- `local_pc_estimate`
- `time`
- `reward_components`
- `episode` summary on final step

## Orbital/ground notes

- The model supports multiple ground stations via `ground_station_thetas` (angles on Earth surface, in radians).
- 3D mode is enabled with `world_dim=3` and uses `(theta, radius, phi)` orbital state.
- Visual projection is controlled by `render_projection` (`2d` map with pygame, or `3d` scene with pyvista).
- For 3D projection, `render_quality` (`ultra_low`, `low`, `medium`, `high`) controls rendering cost vs smoothness.
- Links are line-of-sight constrained: communications do not pass through Earth (`earth_radius`).
- Optional orbital debris clouds are modeled via `enable_debris`; each cloud has drifting position, spread, and density.
- Conjunction pressure is modeled as a local `Pc` proxy from debris density; maneuver actions (`OrbitDown`/`OrbitUp`) reduce risk.
