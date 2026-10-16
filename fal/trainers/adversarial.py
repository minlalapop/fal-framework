from typing import Any
import time

import numpy as np

from fal.evaluation import EvaluationResult

from .base import BaseTrainer


class AdversarialTrainer(BaseTrainer):
    """Train on a clean/adversarial mix."""

    def train(self, data: tuple[Any, Any]) -> Any:
        if self.attack is None:
            raise ValueError("AdversarialTrainer requires an attack.")

        X_train, y_train = data
        attack_started_at = time.perf_counter()
        X_adv = self.attack.generate(
            model=self.model,
            X=X_train,
            y=y_train,
        )
        self.last_attack_generation_train_time = time.perf_counter() - attack_started_at
        X_mix, y_mix = _mix_clean_and_adversarial(X_train, y_train, X_adv, y_train)
        return self.model.fit(X_mix, y_mix, **self.trainer_params)

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        if self.attack is None:
            raise ValueError("AdversarialTrainer requires an attack.")

        X_test, y_test = data
        attack_started_at = time.perf_counter()
        X_adv = self.attack.generate(
            model=self.model,
            X=X_test,
            y=y_test,
        )
        self.last_attack_generation_evaluation_time = time.perf_counter() - attack_started_at
        X_mix, y_mix = _mix_clean_and_adversarial(X_test, y_test, X_adv, y_test)

        y_pred = self.model.predict(X_mix)
        metrics = self.metrics_manager.compute_all(y_mix, y_pred)
        loss = self.loss_manager.compute(self.model, X_mix, y_mix) if self.loss_manager else None

        return EvaluationResult(loss=loss, metrics=metrics)


def _mix_clean_and_adversarial(
    X_clean: Any,
    y_clean: Any,
    X_adv: Any,
    y_adv: Any,
) -> tuple[Any, Any]:
    X_mix = np.concatenate([np.asarray(X_clean), np.asarray(X_adv)], axis=0)
    y_mix = np.concatenate([np.asarray(y_clean), np.asarray(y_adv)], axis=0)
    return X_mix, y_mix
