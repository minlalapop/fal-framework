
"""Mixed trainer for quick clean/attack/adversarial comparison."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

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
        adversarial_model: Any = None,
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
        self.adversarial_model = (
            adversarial_model if adversarial_model is not None else model
        )
        self.adversarial_trainer = AdversarialTrainer(
            model=self.adversarial_model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=self.trainer_params,
        )

    def train(self, data: tuple[Any, Any]) -> Any:
        if self.attack is None:
            raise ValueError("MixedTrainer requires an attack.")

        rng_state = _capture_rng_state(self.model)
        self.clean_trainer.train(data)
        _restore_rng_state(rng_state, self.adversarial_model)
        result = self.adversarial_trainer.train(data)

        self.last_attack_generation_train_time = (
            self.adversarial_trainer.last_attack_generation_train_time
        )
        return result

    def evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        rng_state = _capture_rng_state(self.model)
        clean_result = self.clean_trainer.evaluate(data)
        _restore_rng_state(rng_state, self.model)
        attack_result = self.attack_trainer.evaluate(data)
        _restore_rng_state(rng_state, self.adversarial_model)
        trained_result = self.adversarial_trainer.evaluate(data)

        self.last_attack_generation_evaluation_time = (
            self.attack_trainer.last_attack_generation_evaluation_time
            + self.adversarial_trainer.last_attack_generation_evaluation_time
        )

        metrics = {}
        metrics.update(clean_result.metrics)
        metrics.update(attack_result.metrics)
        metrics.update(trained_result.metrics)

        # This remains AdversarialTrainer's 50/50 clean/adversarial loss.
        return EvaluationResult(loss=trained_result.loss, metrics=metrics)


def _capture_rng_state(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch = getattr(model, "_torch", None)
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], model: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch = getattr(model, "_torch", None)
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
