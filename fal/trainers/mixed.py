"""Mixed trainer for quick clean/attack/adversarial comparison."""

from __future__ import annotations

from typing import Any

from fal.evaluation import EvaluationResult

from .adversarial import AdversarialTrainer
from .attack_only import AttackOnlyTrainer
from .base import BaseTrainer
from .clean import CleanTrainer


class MixedTrainer(BaseTrainer):
    """Run clean training, adversarial training, and grouped evaluation."""

    def __init__(
        self,
        model: Any,
        attack: Any = None,
        metrics_manager: Any = None,
        loss_manager: Any = None,
        trainer_params: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=trainer_params,
        )
        self.clean_trainer = CleanTrainer(
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=self.trainer_params,
        )
        self.attack_trainer = AttackOnlyTrainer(
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params={**self.trainer_params, "clean_fit": False},
        )
        self.adversarial_trainer = AdversarialTrainer(
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=self.trainer_params,
        )

    def train(self, data: tuple[Any, Any]) -> Any:
        if self.attack is None:
            raise ValueError("MixedTrainer requires an attack.")

        self.clean_trainer.train(data)
        return self.adversarial_trainer.train(data)

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        clean_result = self.clean_trainer.evaluate(data)
        attack_result = self.attack_trainer.evaluate(data)
        trained_result = self.adversarial_trainer.evaluate(data)

        metrics = {}
        metrics.update(_prefix_metrics("clean", clean_result.metrics))
        metrics.update(_prefix_metrics("attack", attack_result.metrics))
        metrics.update(_prefix_metrics("trained", trained_result.metrics))

        return EvaluationResult(loss=trained_result.loss, metrics=metrics)


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{name}": value
        for name, value in metrics.items()
    }
