"""Factory for Flower strategies."""

from __future__ import annotations

from typing import Any

from fal.fl.flower.flower_strategy.fedadam_strategy import FedAdamStrategyAdapter
from fal.fl.flower.flower_strategy.fedavg_strategy import FedAvgStrategyAdapter
from fal.fl.flower.flower_strategy.fedprox_strategy import FedProxStrategyAdapter
from fal.fl.flower.flower_strategy.flower_strategy_adapter import StrategyAdapter


class StrategyFactory:
    """Create Flower strategy adapters."""

    _registry = {
        "fedadam": FedAdamStrategyAdapter,
        "fedavg": FedAvgStrategyAdapter,
        "fedprox": FedProxStrategyAdapter,
    }

    @classmethod
    def register(cls, name: str, strategy_cls: type[StrategyAdapter]) -> None:
        cls._registry[cls._normalize_name(name)] = strategy_cls

    @classmethod
    def create(
        cls,
        fl_algorithm: str = "fedavg",
        fl_params: dict[str, Any] | None = None,
    ) -> StrategyAdapter:
        fl_algorithm = cls._normalize_name(fl_algorithm)
        params = fl_params or {}
        strategy_cls = cls._registry.get(fl_algorithm)

        if strategy_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported FL strategy: {fl_algorithm}. Supported strategies: {supported}.")

        return strategy_cls(params)

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("FL algorithm must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
