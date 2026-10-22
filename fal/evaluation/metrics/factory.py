"""Factory for metric managers."""

from __future__ import annotations

from typing import Any

from fal.evaluation.metrics.base import BaseMetric
from fal.evaluation.metrics.manager import MetricsManager
from fal.evaluation.metrics.metrics import (
    AUCMetric,
    AccuracyMetric,
    ClassificationReportMetric,
    ConfusionMatrixMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)


class MetricFactory:
    """Create metric objects from user config."""

    _registry = {
        "accuracy": AccuracyMetric,
        "f1": F1Metric,
        "precision": PrecisionMetric,
        "recall": RecallMetric,
        "auc": AUCMetric,
        "roc_auc": AUCMetric,
        "confusion_matrix": ConfusionMatrixMetric,
        "classification_report": ClassificationReportMetric,
    }
    REGISTRY = _registry

    @classmethod
    def register(cls, name: str, metric_cls: type[BaseMetric]) -> None:
        if not isinstance(metric_cls, type) or not issubclass(metric_cls, BaseMetric):
            raise TypeError("Registered metrics must inherit from BaseMetric.")
        cls._registry[cls._normalize_name(name)] = metric_cls

    @classmethod
    def create(cls, metrics_config: Any) -> MetricsManager:
        configs = _as_list(metrics_config)
        metrics = [cls.create_metric(config) for config in configs]
        return MetricsManager(metrics)

    @classmethod
    def create_metric(cls, config: Any) -> BaseMetric:
        if isinstance(config, BaseMetric):
            return config

        if isinstance(config, str):
            metric_name = cls._normalize_name(config)
            metric_params: dict[str, Any] = {}
        elif isinstance(config, dict):
            metric_name = cls._normalize_name(config.get("name") or config.get("metric"))
            metric_params = {
                key: value
                for key, value in config.items()
                if key not in {"name", "metric"}
            }
        else:
            raise TypeError("Metric config must be a string, dict, or BaseMetric instance.")

        metric_cls = cls._registry.get(metric_name)
        if metric_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported metric `{metric_name}`. Supported: {supported}.")

        return metric_cls(**metric_params)

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

    @staticmethod
    def _normalize_name(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Metric name must be a non-empty string.")
        return value.strip().lower().replace("-", "_")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return ["accuracy"]
    if isinstance(value, (str, dict, BaseMetric)):
        return [value]
    return list(value)
