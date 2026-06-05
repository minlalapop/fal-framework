"""Base attack interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fal.adv.backend_base import AttackBackend


class BaseAttack(ABC):
    """Base interface for logical attacks."""

    def __init__(self, backend: AttackBackend, params: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.params = params or {}

    @abstractmethod
    def generate(self, model: Any, X: Any, y: Any | None = None) -> Any:
        """Generate adversarial examples."""
