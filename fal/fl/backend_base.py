"""Federated learning backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FLBackend(ABC):
    """Interface implemented by concrete FL framework backends."""

    @abstractmethod
    def start(
        self,
        model: Any,
        trainer: Any,
        clients_data: list,
        test_data: Any,
        strategy: Any,
        num_rounds: int = 1,
        verbose: int = 1,
        run_config: dict[str, Any] | None = None,
    ) -> Any:
        """Start a federated learning run."""
