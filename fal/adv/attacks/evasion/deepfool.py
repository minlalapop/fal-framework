"""DeepFool attack."""

from __future__ import annotations

from typing import Any

from fal.adv.attacks.base import BaseAttack


class DeepFool(BaseAttack):
    """DeepFool attack."""

    def generate(self, model: Any, X: Any, y: Any | None = None) -> Any:
        return self.backend.generate("deepfool", model, X, y, self.params)
