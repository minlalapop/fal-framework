"""Metric manager."""

from __future__ import annotations

from typing import Any, Iterable

from fal.evaluation.metrics.base import BaseMetric


class MetricsManager:
    """Compute a collection of metrics with one call."""

    def __init__(self, metrics: Iterable[BaseMetric]) -> None:
        self.metrics = list(metrics)
        if not self.metrics:
            raise ValueError("MetricsManager requires at least one metric.")

    def compute_all(self, y_true: Any, y_pred: Any) -> dict[str, Any]:
        return {metric.name: metric.compute(y_true, y_pred) for metric in self.metrics}
