"""Client splitter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ClientSplitter(ABC):
    """Interface for splitting a dataset across federated clients."""

    @abstractmethod
    def split(self, X: Any, y: Any, num_clients: int) -> list[tuple[Any, Any]]:
        """Split ``X, y`` into one dataset per client."""
