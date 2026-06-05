"""Flower parameter adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FlowerParameterAdapter(ABC):
    """Interface for converting model parameters to/from Flower arrays."""

    @abstractmethod
    def get_parameters(self, model: Any) -> list:
        """Extract model parameters."""

    @abstractmethod
    def set_parameters(self, model: Any, parameters: list) -> None:
        """Load model parameters."""

    @abstractmethod
    def supports(self, model: Any) -> bool:
        """Return whether this adapter supports the model."""
