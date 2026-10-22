"""Experiment orchestration."""

from typing import Any


class Experiment:
    """Object that encapsulates and runs all FAL components."""

    def __init__(
        self,
        dataset_manager: Any,
        model: Any,
        trainer: Any,
        metrics_manager: Any,
        fl_system: Any,
        attack: Any = None,
        verbose: int = 1,
        config: Any = None,
    ):
        self.dataset_manager = dataset_manager
        self.model = model
        self.trainer = trainer
        self.metrics_manager = metrics_manager
        self.fl_system = fl_system
        self.attack = attack
        self.verbose = verbose
        self.config = config

    def execute(self) -> Any:
        """
        Execute the full Federated Adversarial Learning pipeline.
        """
        self.dataset_manager.prepare()

        clients_data = self.dataset_manager.get_clients_data()
        test_data = self.dataset_manager.get_test_data()

        results = self.fl_system.start(
            model=self.model,
            trainer=self.trainer,
            clients_data=clients_data,
            test_data=test_data,
        )
        if hasattr(results, "config"):
            results.config = self.config
        if callable(getattr(results, "save_outputs", None)):
            results.save_outputs(save_model=self._should_save_model())

        if self.verbose >= 1 and callable(getattr(results, "print_details", None)):
            results.print_details()

        return results

    def _should_save_model(self) -> bool:
        if type(self.trainer).__name__ != "AttackOnlyTrainer":
            return True

        trainer_params = getattr(self.trainer, "trainer_params", {})
        return bool(trainer_params.get("clean_fit", False))
