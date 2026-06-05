from typing import Any

from .base import BaseTrainer


class CleanTrainer(BaseTrainer):
    """Clean local training for FL clients."""

    def train(self, data: tuple[Any, Any]) -> Any:
        X_train, y_train = data
        return self.model.fit(X_train, y_train, **self.trainer_params)
