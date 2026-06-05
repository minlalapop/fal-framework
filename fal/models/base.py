"""Common model interface used by the FAL framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Minimal interface shared by all model adapters.

    Flower-specific parameter exchange is intentionally kept outside this
    interface. ``BaseModel`` only represents model training and inference.
    """

    @property
    @abstractmethod
    def framework(self) -> str:
        """Return the backend framework name, for example ``sklearn`` or ``pytorch``."""

    @abstractmethod
    def fit(self, X: Any, y: Any, **params: Any) -> "BaseModel":
        """Train the wrapped model."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return predictions for ``X``."""

    @abstractmethod
    def get_raw_model(self) -> Any:
        """Return the original user-provided model object."""
