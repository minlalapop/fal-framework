from typing import Any

from fal.evaluation import EvaluationResult

from .base import BaseTrainer


class CleanTrainer(BaseTrainer):
    """Clean local training for FL clients."""

    def train(self, data: tuple[Any, Any]) -> Any:
        X_train, y_train = data
        return self.model.fit(X_train, y_train, **self.trainer_params)

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        result = super().evaluate(data)
        return EvaluationResult(
            loss=result.loss,
            metrics={
                f"clean_{name}": value
                for name, value in result.metrics.items()
            },
        )
