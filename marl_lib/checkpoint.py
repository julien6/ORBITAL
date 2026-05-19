from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class PolicyCheckpoint:
    @staticmethod
    def save(path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
        return torch.load(path, map_location=map_location or "cpu")
