"""Build FAL experiment components from config."""

from __future__ import annotations

from fal.adv import AttackBackendFactory, AttackFactory
from fal.dataset import (
    ArrayDatasetLoader,
    BuiltinDatasetLoader,
    CustomCSVDatasetLoader,
    DatasetManager,
    IIDSplitter,
    NonIIDSplitter,
)
from fal.experiment import Experiment
from fal.fl import FLBackendFactory, FLSystem
from fal.fl.flower.flower_strategy import StrategyFactory
from fal.evaluation.losses import LossFactory
from fal.evaluation.metrics import MetricFactory
from fal.models import ModelAdapterFactory
from fal.trainers import TrainerFactory


class CoreBuilder:
    """Construct all internal components for an experiment."""

    def __init__(self, config):
        self.config = config

    def build(self) -> Experiment:
        dataset_manager = self._build_dataset_manager()
        model = self._build_model()
        self._validate_model_fl_compatibility(model)
        metrics_manager = self._build_metrics()
        loss_manager = self._build_loss()
        attack = self._build_attack()
        trainer = self._build_trainer(
            model=model,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            attack=attack,
        )

        fl_system = self._build_fl_system()

        return Experiment(
            dataset_manager=dataset_manager,
            model=model,
            trainer=trainer,
            metrics_manager=metrics_manager,
            attack=attack,
            fl_system=fl_system,
            verbose=self.config.verbose,
        )

    def _build_dataset_manager(self):
        return DatasetManager(
            loader=self._build_dataset_loader(),
            splitter=self._build_client_splitter(),
            num_clients=self.config.num_clients,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
        )

    def _build_dataset_loader(self):
        if self.config.X is not None or self.config.y is not None:
            return ArrayDatasetLoader(self.config.X, self.config.y)

        if isinstance(self.config.dataset, str) and self.config.dataset.lower().endswith(".csv"):
            return CustomCSVDatasetLoader(self.config.dataset, self.config.target)

        return BuiltinDatasetLoader(self.config.dataset, **self.config.dataset_params)

    def _build_client_splitter(self):
        if self.config.iid:
            return IIDSplitter(shuffle_data=self.config.shuffle, random_state=self.config.random_state)
        return NonIIDSplitter(
            shards_per_client=self.config.shards_per_client,
            shuffle_data=self.config.shuffle,
            random_state=self.config.random_state,
        )

    def _build_model(self):
        return ModelAdapterFactory.create(
            model=self.config.model,
            model_params=self.config.model_params,
        )

    def _build_metrics(self):
        return MetricFactory.create(self.config.metrics)

    def _build_loss(self):
        return LossFactory.create(self.config.loss)

    def _build_attack(self):
        if self.config.attack_type is None:
            return None

        backend = AttackBackendFactory.create(
            self.config.attack_backend
        )

        return AttackFactory.create(
            attack_type=self.config.attack_type,
            attack_params=self.config.attack_params,
            backend=backend,
        )

    def _build_trainer(self, model, metrics_manager, loss_manager, attack):
        trainer_params = dict(self.config.trainer_params)
        if self.config.trainer_type == "attack_only" and self._attack_only_needs_clean_fit(model):
            trainer_params.setdefault("clean_fit", True)

        return TrainerFactory.create(
            trainer_type=self.config.trainer_type,
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=trainer_params,
        )

    def _validate_model_fl_compatibility(self, model):
        if self.config.fl_algorithm == "fedprox" and getattr(model, "framework", None) == "sklearn":
            raise ValueError(
                "FedProx proximal training is not supported for sklearn models. "
                "Use `fl_algorithm='fedavg'` or a PyTorch model for FedProx."
            )

    def _attack_only_needs_clean_fit(self, model):
        if self.config.trainer_type != "attack_only":
            return False

        if getattr(model, "framework", None) == "pytorch":
            return bool(self.config.model_params)

        if getattr(model, "framework", None) == "sklearn":
            return not _is_fitted_sklearn_model(model.get_raw_model())

        return False

    def _build_fl_system(self):
        strategy = StrategyFactory.create(
            fl_algorithm=self.config.fl_algorithm,
            fl_params=self.config.fl_params,
        )

        backend = FLBackendFactory.create(
            self.config.fl_backend
        )

        return FLSystem(
            backend=backend,
            strategy=strategy,
            num_rounds=self.config.num_rounds,
            verbose=self.config.verbose,
            run_config=self._build_run_config(),
        )

    def _build_run_config(self):
        return {
            "fl_backend": self.config.fl_backend,
            "fl_algorithm": self.config.fl_algorithm,
            "trainer_type": self.config.trainer_type,
            "attack": self._build_attack_display_config(),
            "dataset": self._display_dataset(),
            "model": type(self.config.model).__name__,
            "metrics": [self._metric_name(metric) for metric in self.config.metrics],
            "loss": self._loss_name(),
            "random_state": self.config.random_state,
            "deterministic": self.config.deterministic,
        }

    def _build_attack_display_config(self):
        if self.config.attack_type is None:
            return None
        return {
            "type": self.config.attack_type,
            "params": dict(self.config.attack_params),
        }

    def _display_dataset(self):
        if self.config.dataset is not None:
            return self.config.dataset
        return "array"

    def _metric_name(self, metric):
        if isinstance(metric, str):
            return metric
        if isinstance(metric, dict):
            return metric.get("name") or metric.get("metric")
        return type(metric).__name__

    def _loss_name(self):
        if self.config.loss is None:
            return None
        if isinstance(self.config.loss, str):
            return self.config.loss
        if isinstance(self.config.loss, dict):
            return self.config.loss.get("name") or self.config.loss.get("loss")
        return type(self.config.loss).__name__


FALBuilder = CoreBuilder


def _is_fitted_sklearn_model(model):
    return any(
        name.endswith("_") and not name.startswith("__")
        for name in vars(model)
    )
