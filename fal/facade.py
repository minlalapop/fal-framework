"""Public FAL facade."""

from __future__ import annotations

from typing import Any

from fal.builder import CoreBuilder
from fal.config import FALConfig


class FAL:
    """Public entrypoint for Federated Adversarial Learning experiments."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = FALConfig(**kwargs)

    def run(self) -> Any:
        self.config.validate()
        experiment = CoreBuilder(self.config).build()
        return experiment.execute()
