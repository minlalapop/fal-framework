"""Factory for evaluation loss managers."""

from __future__ import annotations

from typing import Any

from fal.evaluation.losses.base import BaseEvaluationLoss
from fal.evaluation.losses.losses import LogLoss, MSELoss, ModelLoss
from fal.evaluation.losses.manager import EvaluationLossManager


class EvaluationLossFactory:
    """Create evaluation loss objects from user config."""

    _registry = {
        "log_loss": LogLoss,
        "model_loss": ModelLoss,
        "mse": MSELoss,
        "mean_squared_error": MSELoss,
    }

    @classmethod
    def register(cls, name: str, loss_cls: type[BaseEvaluationLoss]) -> None:
        cls._registry[cls._normalize_name(name)] = loss_cls

    @classmethod
    def create(cls, loss_config: Any = None) -> EvaluationLossManager:
        if loss_config is None:
            return EvaluationLossManager()

        loss = cls.create_loss(loss_config)
        return EvaluationLossManager(loss)

    @classmethod
    def create_loss(cls, config: Any) -> BaseEvaluationLoss:
        if isinstance(config, BaseEvaluationLoss):
            return config

        if isinstance(config, str):
            loss_name = cls._normalize_name(config)
            loss_params: dict[str, Any] = {}
        elif isinstance(config, dict):
            loss_name = cls._normalize_name(config.get("name") or config.get("loss"))
            loss_params = {
                key: value
                for key, value in config.items()
                if key not in {"name", "loss"}
            }
        else:
            raise TypeError("Loss config must be a string, dict, BaseEvaluationLoss instance, or None.")

        loss_cls = cls._registry.get(loss_name)
        if loss_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported loss `{loss_name}`. Supported: {supported}.")

        return loss_cls(**loss_params)

    @staticmethod
    def _normalize_name(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Loss name must be a non-empty string.")
        return value.strip().lower().replace("-", "_")


LossFactory = EvaluationLossFactory
