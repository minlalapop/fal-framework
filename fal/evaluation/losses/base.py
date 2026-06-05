"""Evaluation loss interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluationLoss(ABC):
    """Base interface for losses computed during evaluation."""

    name: str

    @abstractmethod
    def compute(self, model: Any, X: Any, y_true: Any) -> float:
        """Compute the evaluation loss."""


BaseLoss = BaseEvaluationLoss
