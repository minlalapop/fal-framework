"""FGSM attack."""

from __future__ import annotations

from typing import Any

from fal.adv.attacks.base import BaseAttack


class FGSM(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def generate(self, model: Any, X: Any, y: Any | None = None) -> Any:
        return self.backend.generate("fgsm", model, X, y, self.params)
