"""Unified federated learning system facade."""

from __future__ import annotations

from typing import Any

from fal.fl.backend_base import FLBackend


class FLSystem:
    """Small facade that hides the concrete FL framework backend."""

    def __init__(
        self,
        backend: FLBackend,
        strategy: Any,
        num_rounds: int = 1,
        verbose: int = 1,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.verbose = verbose
        self.run_config = run_config or {}

    def start(self, model: Any, trainer: Any, clients_data: list, test_data: Any) -> Any:
        return self.backend.start(
            model=model,
            trainer=trainer,
            clients_data=clients_data,
            test_data=test_data,
            strategy=self.strategy,
            num_rounds=self.num_rounds,
            verbose=self.verbose,
            run_config=self.run_config,
        )
