from abc import ABC, abstractmethod
from typing import Any

from fal.evaluation import EvaluationResult


class BaseTrainer(ABC):
    def __init__(
        self,
        model: Any,
        attack: Any = None,
        metrics_manager: Any = None,
        loss_manager: Any = None,
        trainer_params: dict | None = None,
    ) -> None:
        self.model = model
        self.attack = attack
        self.metrics_manager = metrics_manager
        self.loss_manager = loss_manager
        self.trainer_params = trainer_params or {}
        self.last_attack_generation_train_time = 0.0
        self.last_attack_generation_evaluation_time = 0.0

    @abstractmethod
    def train(self, data: tuple[Any, Any]) -> Any:
        """Train the wrapped model according to the trainer strategy."""

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        X_test, y_test = data
        y_pred = self.model.predict(X_test)
        metrics = self.metrics_manager.compute_all(y_test, y_pred)
        loss = self.loss_manager.compute(self.model, X_test, y_test) if self.loss_manager else None
        return EvaluationResult(loss=loss, metrics=metrics)
