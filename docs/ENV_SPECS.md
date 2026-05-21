# ORBITAL Environment Specs

## Operational Goal (Intuitive)

The team goal is to maximize mission value delivered to ground while preserving fleet viability.

In practice, this means balancing four pressures at every step:

1. acquisition (discover known orbital observation tasks and collect useful data),
2. delivery (route buffered data to ground directly or through satellites),
3. stabilization (preserve health, energy, connectivity, and cyber integrity),
4. conjunction safety (limit debris exposure and collision risk).

Important modeling choice: `Observe` and `Relay` are intentionally decoupled.
Servicing a task creates buffered data, but value is only realized when data is later delivered.
This is what makes myopic policies fail in long horizons.

## Observation (fixed-size vector)

Each agent receives a `np.float32` vector of size **20**:

1. normalized self energy `[0,1]`
2. normalized self health `[0,1]`
3. normalized orbital angle `theta/(2π)` in `[0,1]`
4. normalized instantaneous orbital radius in `[0,1]`
5. normalized derived orbital latitude `phi` in 3D, or `0` in 2D
6. sunlight indicator
7. direct ground contact indicator
8. estimated route to ground indicator
9. normalized local communication degree
10. normalized buffered data
11. normalized remaining buffer capacity
12. normalized known nearby task count
13. normalized known nearby task priority
14. normalized local debris density
15. local conjunction-risk proxy (`Pc` estimate in `[0,1]`)
16. compromised indicator
17. suspicious/scan indicator
18. jammed indicator
19. fraction of compromised neighbors
20. alive satellite fraction, negated if the last action was malware-forced

> `obs_mode` is intentionally not exposed: ORBITAL uses fixed-size vectors only.

## Action space

`Discrete(8)`:

- `0 OBS`
- `1 REL_GRN` (bidirectional satellite-ground relay)
- `2 REL_SAT` (bidirectional satellite-satellite relay)
- `3 DN` (transfer to a lower orbit)
- `4 UP` (transfer to a higher orbit)
- `5 PWR`
- `6 SCAN`
- `7 IDLE`

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
- `health`
- `alive`
- `compromised`
- `malware_awake`
- `jammed`
- `last_action_forced`
- `local_degree`
- `buffered_data`
- `known_tasks`
- `theta`
- `phi`
- `radius`
- `semi_major_axis`
- `eccentricity`
- `mean_anomaly`
- `arg_periapsis`
- `inclination`
- `raan`
- `sunlight`
- `ground_contact`
- `ground_route`
- `local_debris_density`
- `local_pc_estimate`
- `last_executed_action`
- episode counters such as `delivered_total`, `observed_total`, `knowledge_shared`, `jam_count`, and `forced_actions`
- `time`
- `reward_components`
- `episode` summary on final step

## Orbital/ground notes

- The model supports multiple ground stations via `ground_station_thetas` (angles on Earth surface, in radians).
- All moving bodies are propagated from Keplerian elements: semi-major axis, eccentricity, mean anomaly, argument of periapsis, inclination, and right ascension of the ascending node.
- `theta`, `radius`, and `phi` remain derived spherical coordinates for observations, proximity checks, and rendering.
- 3D mode is enabled with `world_dim=3`; 2D mode uses the same elliptic propagation in the equatorial plane.
- Visual projection is controlled by `render_projection` (`2d` map with pygame, or `3d` scene with pyvista).
- For 3D projection, `render_quality` (`ultra_low`, `low`, `medium`, `high`) controls rendering cost vs smoothness.
- Satellites, observation tasks, and debris clouds advance mean anomaly with mean motion `kepler_constant / semi_major_axis^(3/2)` and solve Kepler's equation for their instantaneous position.
- Links are line-of-sight constrained: communications do not pass through Earth (`earth_radius`).
- Optional orbital debris clouds are modeled via `enable_debris`; each cloud has an orbital position, spread, and density.
- Conjunction pressure is modeled as a local `Pc` proxy from debris density; maneuver actions (`OrbitDown`/`OrbitUp`) reduce risk.
