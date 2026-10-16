"""Public FAL facade."""

from __future__ import annotations

import time
from typing import Any

from fal.builder import CoreBuilder
from fal.config import FALConfig
from fal.utils.randomness import seed_everything


class FAL:
    """Public entrypoint for Federated Adversarial Learning experiments."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = FALConfig(**kwargs)

    def run(self) -> Any:
        started_at = time.perf_counter()
        self.config.validate()
        seed_everything(
            self.config.random_state,
            deterministic=self.config.deterministic,
        )
        experiment = CoreBuilder(self.config).build()
        result = experiment.execute()
        timings = getattr(result, "timings", None)
        if isinstance(timings, dict):
            timings["total_execution_time"] = time.perf_counter() - started_at
            if callable(getattr(result, "refresh_timings_output", None)):
                result.refresh_timings_output()
        return result
