"""FedAvg strategy adapter."""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAvg

from fal.fl.flower.flower_strategy.flower_strategy_adapter import StrategyAdapter


class FedAvgStrategyAdapter(StrategyAdapter):
    """Build Flower FedAvg."""

    def __init__(self, fl_params: dict[str, Any] | None = None) -> None:
        super().__init__(fl_params)

    def build_strategy(self) -> Any:
        return FedAvg(**self._params_with_default_aggregation())
