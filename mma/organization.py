from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class RoleRule:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalRule:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    name: str
    rules: list[RoleRule] = field(default_factory=list)


@dataclass
class Goal:
    name: str
    rules: list[GoalRule] = field(default_factory=list)


@dataclass
class AgentAssignment:
    agent: str
    roles: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)


@dataclass
class Organization:
    name: str
    roles: dict[str, Role] = field(default_factory=dict)
    goals: dict[str, Goal] = field(default_factory=dict)
    assignments: dict[str, AgentAssignment] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Organization":
        roles = {
            name: Role(name=raw.get("name", name), rules=[RoleRule(**r) for r in raw.get("rules", [])])
            for name, raw in data.get("roles", {}).items()
        }
        goals = {
            name: Goal(name=raw.get("name", name), rules=[GoalRule(**r) for r in raw.get("rules", [])])
            for name, raw in data.get("goals", {}).items()
        }
        assignments = {
            name: AgentAssignment(
                agent=raw.get("agent", name),
                roles=list(raw.get("roles", [])),
                goals=list(raw.get("goals", [])),
            )
            for name, raw in data.get("assignments", {}).items()
        }
        return cls(
            name=data["name"],
            roles=roles,
            goals=goals,
            assignments=assignments,
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Organization":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def assignment_for(self, agent: str) -> AgentAssignment:
        return self.assignments.get(agent, AgentAssignment(agent=agent))
