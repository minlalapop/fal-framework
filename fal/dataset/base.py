"""Dataset loading interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetLoader(ABC):
    """Interface for objects that load data as ``X, y``."""

    @abstractmethod
    def load(self) -> tuple[Any, Any]:
        """Load features and labels."""
