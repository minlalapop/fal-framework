
from typing import Any
import time

import numpy as np

from fal.adv.attacks.attack_factory import AttackFactory
from fal.evaluation import EvaluationResult

from .base import BaseTrainer


_EVALUATION_PARAM_KEYS = {
    "evaluation_attack_type",
    "evaluation_attack_params",
}


class AdversarialTrainer(BaseTrainer):
    """Train on a clean/adversarial mix."""

    def train(self, data: tuple[Any, Any]) -> Any:
        if self.attack is None:
            raise ValueError("AdversarialTrainer requires an attack.")

        X_train, y_train = data

        # Ne pas transmettre les options d'évaluation à model.fit()
        params = {
            key: value
            for key, value in self.trainer_params.items()
            if key not in _EVALUATION_PARAM_KEYS
        }

        if params.pop("online_adversarial", False):
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

            epochs = int(params.get(
                "local_steps",
                params.get(
                    "epochs",
                    getattr(self.model, "epochs", 1),
                ),
            ))

            batch_size = int(params.get(
                "batch_size",
                getattr(self.model, "batch_size", 32),
            ))

            if epochs < 1 or batch_size < 2:
                raise ValueError("Invalid epochs or batch_size.")

            clean_batch_size = batch_size // 2
            attack_time = 0.0

            for _ in range(epochs):
                indices = np.random.permutation(len(X_train))

                for start in range(0, len(X_train), clean_batch_size):
                    batch_indices = indices[
                        start:start + clean_batch_size
                    ]

                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    attack_started_at = time.perf_counter()

                    X_adv = self.attack.generate(
                        model=self.model,
                        X=X_batch,
                        y=y_batch,
                    )

                    attack_time += (
                        time.perf_counter() - attack_started_at
                    )

                    X_mix, y_mix = _mix_clean_and_adversarial(
                        X_batch, y_batch, X_adv, y_batch
                    )

                    self.model.fit(
                        X_mix,
                        y_mix,
                        **{
                            **params,
                            "local_steps": 1,
                            "epochs": 1,
                            "batch_size": len(y_mix),
                        },
                    )

            self.last_attack_generation_train_time = attack_time
            return self.model

        # Entraînement offline original
        attack_started_at = time.perf_counter()

        X_adv = self.attack.generate(
            model=self.model,
            X=X_train,
            y=y_train,
        )

        self.last_attack_generation_train_time = (
            time.perf_counter() - attack_started_at
        )

        X_mix, y_mix = _mix_clean_and_adversarial(
            X_train, y_train, X_adv, y_train
        )

        return self.model.fit(X_mix, y_mix, **params)

    def _get_evaluation_attack(self) -> Any:
        """
        Return the independent evaluation attack if configured.
        Otherwise, keep using the training attack.
        """
        evaluation_attack = getattr(
            self, "evaluation_attack", None
        )

        if evaluation_attack is not None:
            return evaluation_attack

        evaluation_attack_type = self.trainer_params.get(
            "evaluation_attack_type"
        )

        if evaluation_attack_type is None:
            return self.attack

        evaluation_attack = AttackFactory.create(
            attack_type=evaluation_attack_type,
            attack_params=self.trainer_params.get(
                "evaluation_attack_params", {}
            ),
            backend=self.attack.backend,
        )

        self.evaluation_attack = evaluation_attack
        return evaluation_attack

    def evaluate(
        self,
        data: tuple[Any, Any],
    ) -> EvaluationResult:
        """Evaluate the adversarially trained model on clean and attacked data."""

        if self.attack is None:
            raise ValueError("AdversarialTrainer requires an attack.")

        X_test, y_test = data

        evaluation_attack = self._get_evaluation_attack()

        attack_started_at = time.perf_counter()

        X_adv = evaluation_attack.generate(
            model=self.model,
            X=X_test,
            y=y_test,
        )

        self.last_attack_generation_evaluation_time = (
            time.perf_counter() - attack_started_at
        )

        # Même proportion 50 % clean + 50 % adversarial
        X_mix, y_mix = _mix_clean_and_adversarial(
            X_test, y_test, X_adv, y_test
        )

        # Une seule prédiction sur le mélange
        y_pred = np.asarray(self.model.predict(X_mix))

        # Métriques séparées sur les deux moitiés
        n_clean = len(y_test)

        clean_metrics = self.metrics_manager.compute_all(
            y_test,
            y_pred[:n_clean],
        )

        adversarial_metrics = self.metrics_manager.compute_all(
            y_test,
            y_pred[n_clean:],
        )

        metrics = {
            f"trained_clean_{name}": value
            for name, value in clean_metrics.items()
        }

        metrics.update({
            f"trained_adv_{name}": value
            for name, value in adversarial_metrics.items()
        })

        if self.loss_manager:
            metrics["trained_clean_loss"] = self.loss_manager.compute(
                self.model,
                X_test,
                y_test,
            )
            metrics["trained_adv_loss"] = self.loss_manager.compute(
                self.model,
                X_adv,
                y_test,
            )

        return EvaluationResult(
            metrics=metrics,
        )


def _mix_clean_and_adversarial(
    X_clean: Any,
    y_clean: Any,
    X_adv: Any,
    y_adv: Any,
) -> tuple[Any, Any]:

    X_mix = np.concatenate(
        [np.asarray(X_clean), np.asarray(X_adv)],
        axis=0,
    )

    y_mix = np.concatenate(
        [np.asarray(y_clean), np.asarray(y_adv)],
        axis=0,
    )

    return X_mix, y_mix
