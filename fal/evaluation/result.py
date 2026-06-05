"""Internal evaluation result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Framework-neutral evaluation output."""

    loss: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        values = dict(self.metrics)
        if self.loss is not None:
            values["loss"] = self.loss
        return values
