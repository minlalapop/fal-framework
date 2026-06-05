"""PGD attack."""

from __future__ import annotations

from typing import Any

from fal.adv.attacks.base import BaseAttack


class PGD(BaseAttack):
    """Projected Gradient Descent attack."""

    def generate(self, model: Any, X: Any, y: Any | None = None) -> Any:
        return self.backend.generate("pgd", model, X, y, self.params)
