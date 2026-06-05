"""Attack backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AttackBackend(ABC):
    """Backend capable of generating adversarial examples."""

    @abstractmethod
    def generate(
        self,
        attack_name: str,
        model: Any,
        X: Any,
        y: Any | None,
        params: dict[str, Any],
    ) -> Any:
        """Generate adversarial examples for ``X``."""
