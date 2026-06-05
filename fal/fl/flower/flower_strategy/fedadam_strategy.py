"""FedAdam strategy adapter."""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAdam

from fal.fl.flower.flower_strategy.flower_strategy_adapter import StrategyAdapter


class FedAdamStrategyAdapter(StrategyAdapter):
    """Build Flower FedAdam."""

    def __init__(self, fl_params: dict[str, Any] | None = None) -> None:
        super().__init__(fl_params)

    def build_strategy(self) -> Any:
        params = self._params_with_default_aggregation()
        params.setdefault("initial_parameters", self.initial_parameters)
        return FedAdam(**params)
