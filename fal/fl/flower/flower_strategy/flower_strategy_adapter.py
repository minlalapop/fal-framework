"""Flower strategy adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StrategyAdapter(ABC):
    """Interface for Flower strategy adapters."""

    def __init__(self, fl_params: dict[str, Any] | None = None) -> None:
        self.fl_params = fl_params or {}
        self.initial_parameters = None

    def set_initial_parameters(self, parameters: Any) -> None:
        self.initial_parameters = parameters

    @abstractmethod
    def build_strategy(self) -> Any:
        """Build a Flower strategy object."""

    def _params_with_default_aggregation(self) -> dict[str, Any]:
        params = dict(self.fl_params)
        params.setdefault("evaluate_metrics_aggregation_fn", _weighted_average_metrics)
        return params


def _weighted_average_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    total_examples: dict[str, int] = {}
    artifacts: dict[str, Any] = {}

    for num_examples, client_metrics in metrics:
        for name, value in client_metrics.items():
            if not isinstance(value, (int, float)):
                artifacts.setdefault(name, value)
                continue
            totals[name] = totals.get(name, 0.0) + float(value) * num_examples
            total_examples[name] = total_examples.get(name, 0) + num_examples

    averaged = {
        name: total / total_examples[name]
        for name, total in totals.items()
        if total_examples[name] > 0
    }
    return {**averaged, **artifacts}
