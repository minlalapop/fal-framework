"""FedProx strategy adapter."""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedProx

from fal.fl.flower.flower_strategy.flower_strategy_adapter import StrategyAdapter


class FedProxStrategyAdapter(StrategyAdapter):
    """Build Flower FedProx."""

    def __init__(self, fl_params: dict[str, Any] | None = None) -> None:
        super().__init__(fl_params)

    def build_strategy(self) -> Any:
        params = self._params_with_default_aggregation()
        if "proximal_mu" not in params:
            raise ValueError(
                "FedProx requires `fl_params={'proximal_mu': value}`. "
                "Example: `fl_params={'proximal_mu': 0.1}`."
            )
        params.setdefault("initial_parameters", self.initial_parameters)
        return FedProx(**params)
