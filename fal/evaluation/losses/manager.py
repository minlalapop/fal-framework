"""Evaluation loss manager."""

from __future__ import annotations

from typing import Any

from fal.evaluation.losses.base import BaseEvaluationLoss


class EvaluationLossManager:
    """Compute an optional evaluation loss."""

    def __init__(self, loss: BaseEvaluationLoss | None = None) -> None:
        self.loss = loss

    def compute(self, model: Any, X: Any, y_true: Any) -> float | None:
        if self.loss is None:
            return None
        return self.loss.compute(model, X, y_true)


LossManager = EvaluationLossManager
