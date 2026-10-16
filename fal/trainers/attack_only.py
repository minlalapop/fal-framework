from typing import Any
import time

from fal.evaluation import EvaluationResult

from .base import BaseTrainer


class AttackOnlyTrainer(BaseTrainer):
    """
    Evaluate the model under adversarial attack.
    """

    def train(self, data: tuple[Any, Any]) -> Any:
        if self.trainer_params.get("clean_fit", False):
            X_train, y_train = data
            return self.model.fit(X_train, y_train, **self.trainer_params)

        return self.model

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        if self.attack is None:
            raise ValueError("AttackOnlyTrainer requires an attack.")

        X_test, y_test = data

        attack_started_at = time.perf_counter()
        X_adv = self.attack.generate(
            model=self.model,
            X=X_test,
            y=y_test,
        )
        self.last_attack_generation_evaluation_time = time.perf_counter() - attack_started_at

        y_pred = self.model.predict(X_adv)
        metrics = self.metrics_manager.compute_all(y_test, y_pred)
        loss = self.loss_manager.compute(self.model, X_adv, y_test) if self.loss_manager else None

        return EvaluationResult(loss=loss, metrics=metrics)
