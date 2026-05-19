from __future__ import annotations

from collections.abc import Callable
from typing import Any


RoleRuleFn = Callable[[list[dict[str, Any]], dict[str, Any], str, dict[str, Any]], list[int]]
GoalRuleFn = Callable[
    [list[dict[str, Any]], dict[str, Any], dict[str, int], dict[str, int], dict[str, float], dict[str, Any], str, dict[str, Any]],
    float,
]

ROLE_RULES: dict[str, RoleRuleFn] = {}
GOAL_RULES: dict[str, GoalRuleFn] = {}


def register_role_rule(name: str, fn: RoleRuleFn | None = None):
    def _decorator(func: RoleRuleFn) -> RoleRuleFn:
        ROLE_RULES[name] = func
        return func

    if fn is not None:
        return _decorator(fn)
    return _decorator


def register_goal_rule(name: str, fn: GoalRuleFn | None = None):
    def _decorator(func: GoalRuleFn) -> GoalRuleFn:
        GOAL_RULES[name] = func
        return func

    if fn is not None:
        return _decorator(fn)
    return _decorator


def get_role_rule(name: str) -> RoleRuleFn:
    if name not in ROLE_RULES:
        raise KeyError(f"Unknown MMA role rule: {name}")
    return ROLE_RULES[name]


def get_goal_rule(name: str) -> GoalRuleFn:
    if name not in GOAL_RULES:
        raise KeyError(f"Unknown MMA goal rule: {name}")
    return GOAL_RULES[name]
