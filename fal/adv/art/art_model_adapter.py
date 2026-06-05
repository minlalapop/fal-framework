"""ART model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ARTModelAdapter(ABC):
    """Interface for converting a model adapter into an ART estimator."""

    @abstractmethod
    def create_estimator(
        self,
        model: Any,
        config: dict[str, Any],
        X: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        """Create an ART estimator."""
