"""Metric interface used by trainers and evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Base interface for all FAL metrics."""

    name: str

    @abstractmethod
    def compute(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the metric value."""
