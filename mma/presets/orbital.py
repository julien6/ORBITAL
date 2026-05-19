from __future__ import annotations

from typing import Any

from mma.organization import AgentAssignment, Goal, GoalRule, Organization, Role, RoleRule
from mma.registry import register_goal_rule, register_role_rule

OBS = 0
REL_GRN = 1
REL_SAT = 2
DN = 3
UP = 4
PWR = 5
SCAN = 6
IDLE = 7


def _agent_obs(joint: dict[str, Any], agent: str):
    return joint.get("obs", {}).get(agent)


@register_role_rule("orbital.acquirer")
def acquirer_rule(history, joint, agent, params):
    obs = _agent_obs(joint, agent)
    allowed = {OBS, REL_GRN, PWR, SCAN, IDLE}
    if obs is None:
        return sorted(allowed)
    energy, buffer, known_local, debris, compromised = float(obs[0]), float(obs[9]), float(obs[11]), float(obs[14]), float(obs[15])
    if energy < params.get("min_obs_energy", 0.18) or buffer >= params.get("max_obs_buffer", 0.92) or known_local <= 0.0:
        allowed.discard(OBS)
    if debris >= params.get("debris_escape", 0.55):
        allowed.update({DN, UP})
    if compromised > 0.0:
        allowed.add(SCAN)
    return sorted(allowed)


@register_role_rule("orbital.ground_courier")
def ground_courier_rule(history, joint, agent, params):
    obs = _agent_obs(joint, agent)
    allowed = {REL_GRN, REL_SAT, PWR, SCAN, IDLE}
    if obs is None:
        return sorted(allowed)
    energy, ground_contact, route, buffer, debris = float(obs[0]), float(obs[6]), float(obs[7]), float(obs[9]), float(obs[14])
    if buffer <= params.get("min_buffer", 0.05) and ground_contact <= 0.0 and route <= 0.0:
        allowed.add(OBS)
    if energy < params.get("min_relay_energy", 0.12):
        allowed.discard(REL_GRN)
        allowed.discard(REL_SAT)
        allowed.add(PWR)
    if debris >= params.get("debris_escape", 0.60):
        allowed.update({DN, UP})
    return sorted(allowed)


@register_role_rule("orbital.sat_router")
def sat_router_rule(history, joint, agent, params):
    obs = _agent_obs(joint, agent)
    allowed = {REL_SAT, REL_GRN, PWR, SCAN, IDLE}
    if obs is None:
        return sorted(allowed)
    energy, health, degree, debris, low_or_high = float(obs[0]), float(obs[1]), float(obs[8]), float(obs[14]), float(obs[3])
    if degree <= 0.0:
        allowed.add(REL_GRN)
    if energy < params.get("min_route_energy", 0.15):
        allowed.discard(REL_SAT)
        allowed.discard(REL_GRN)
    if health < params.get("low_health", 0.35) or debris > params.get("debris_escape", 0.45):
        allowed.update({DN, UP})
    if low_or_high < 0.08:
        allowed.add(UP)
    elif low_or_high > 0.92:
        allowed.add(DN)
    return sorted(allowed)


@register_goal_rule("orbital.acquire_useful_data")
def acquire_useful_data(history, joint, sampled, executed, rewards, infos, agent, params):
    obs = _agent_obs(joint, agent)
    act = int(executed.get(agent, IDLE))
    if obs is None or act != OBS:
        return 0.0
    known_local = float(obs[11])
    capacity = float(obs[10])
    if known_local > 0.0 and capacity > 0.05:
        return float(params.get("bonus", 0.08))
    return -float(params.get("waste_penalty", 0.04))


@register_goal_rule("orbital.deliver_fast")
def deliver_fast(history, joint, sampled, executed, rewards, infos, agent, params):
    info = infos.get(agent, {})
    act = int(executed.get(agent, IDLE))
    bonus = 0.0
    components = info.get("reward_components", {})
    if act == REL_GRN and info.get("ground_contact", False):
        bonus += float(params.get("ground_bonus", 0.08))
    if act == REL_SAT:
        bonus += float(params.get("sat_bonus", 0.04))
    if float(info.get("buffered_data", 0.0)) >= float(params.get("saturation_buffer", 4.5)):
        bonus -= float(params.get("saturation_penalty", 0.04))
    bonus += float(components.get("delivery", 0.0)) * float(params.get("delivery_scale", 0.03))
    return bonus


@register_goal_rule("orbital.stabilize_fleet")
def stabilize_fleet(history, joint, sampled, executed, rewards, infos, agent, params):
    info = infos.get(agent, {})
    act = int(executed.get(agent, IDLE))
    bonus = 0.0
    if act == SCAN and (info.get("compromised", False) or info.get("malware_awake", False)):
        bonus += float(params.get("scan_bonus", 0.06))
    if act == PWR and info.get("sunlight", False):
        bonus += float(params.get("power_bonus", 0.03))
    if info.get("jammed", False) or info.get("last_action_forced", False):
        bonus -= float(params.get("cyber_penalty", 0.05))
    if float(info.get("local_pc_estimate", 0.0)) > float(params.get("pc_threshold", 0.45)) and act not in {DN, UP}:
        bonus -= float(params.get("debris_penalty", 0.04))
    return bonus


def make_orbital_basic_organization(agent_names: list[str] | None = None) -> Organization:
    if agent_names is None:
        agent_names = [f"sat_{i}" for i in range(8)]
    roles = {
        "Acquirer": Role("Acquirer", [RoleRule("orbital.acquirer")]),
        "GroundCourier": Role("GroundCourier", [RoleRule("orbital.ground_courier")]),
        "SatRouter": Role("SatRouter", [RoleRule("orbital.sat_router")]),
    }
    goals = {
        "AcquireUsefulData": Goal("AcquireUsefulData", [GoalRule("orbital.acquire_useful_data")]),
        "DeliverFast": Goal("DeliverFast", [GoalRule("orbital.deliver_fast")]),
        "StabilizeFleet": Goal("StabilizeFleet", [GoalRule("orbital.stabilize_fleet")]),
    }
    assignments = {}
    pattern = [
        (["Acquirer"], ["AcquireUsefulData", "StabilizeFleet"]),
        (["GroundCourier"], ["DeliverFast", "StabilizeFleet"]),
        (["SatRouter"], ["DeliverFast", "StabilizeFleet"]),
    ]
    for idx, agent in enumerate(agent_names):
        roles_i, goals_i = pattern[idx % len(pattern)]
        assignments[agent] = AgentAssignment(agent=agent, roles=list(roles_i), goals=list(goals_i))
    return Organization(
        name="orbital_basic",
        roles=roles,
        goals=goals,
        assignments=assignments,
        metadata={"description": "Acquire--deliver--stabilize starter organization for ORBITAL V2."},
    )
