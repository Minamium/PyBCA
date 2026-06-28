from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    simulator: Any | None = None
    current_step: int = 0
    elapsed_sec: float = 0.0
    event_history: Any | None = None
    rule_history: Any | None = None
    meta: dict[str, Any] | None = None
