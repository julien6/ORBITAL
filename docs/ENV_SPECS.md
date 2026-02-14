# ORBITAL Environment Specs

## Observation (fixed-size vector)

Each agent receives a `np.float32` vector of size **14**:

1. normalized self energy `[0,1]`
2. normalized x position `[0,1]`
3. normalized y position `[0,1]`
4. cyber healthy indicator (one-hot)
5. cyber suspicious indicator (one-hot)
6. cyber compromised indicator (one-hot)
7. normalized local communication degree
8. normalized nearby active task count
9. normalized nearby task priority sum
10. normalized buffered data
11. sunlight phase `sin(2πt/T)`
12. sunlight phase `cos(2πt/T)`
13. fraction of compromised neighbors
14. alive satellite fraction

> `obs_mode` is intentionally not exposed: ORBITAL uses fixed-size vectors only.

## Action space

`Discrete(6)`:

- `0 Observe`
- `1 Relay`
- `2 Move`
- `3 LowPower`
- `4 CyberScan`
- `5 Idle`

## Reward

By default, `reward_mode="shared"` with weighted components:

- `+ task serviced`
- `+ data delivered to ground`
- `- energy consumed`
- `- isolated satellites`
- `- satellite failures`
- `- cyber impact`

Configure via `reward_weights` in constructor.

## Termination / truncation

- truncation when `t >= max_steps`
- termination when mission collapse occurs (all dead or critically few alive)

## Infos

`infos[agent]` includes:

- `energy`
- `compromised`
- `local_degree`
- `buffered_data`
- `time`
- `reward_components`
- `episode` summary on final step
