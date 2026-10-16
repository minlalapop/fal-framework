"""Flower client adapter."""

from __future__ import annotations

import math
import time
from typing import Any

import flwr as fl

from fal.evaluation import EvaluationResult
from fal.fl.flower.metric_codec import encode_metric_value
from fal.utils.randomness import derive_client_seed, seed_everything


FAL_TELEMETRY_PREFIX = "__fal_telemetry__"
FAL_SERVER_ROUND_KEY = "__fal_server_round"


class FlowerClientAdapter:
    """Adapter between Flower client calls and FAL trainers/models."""

    def __init__(
        self,
        model: Any,
        trainer: Any,
        parameter_adapter: Any,
        data: tuple[Any, Any],
        test_data: tuple[Any, Any] | None = None,
        client_id: int = 0,
        random_state: int | None = None,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.parameter_adapter = parameter_adapter
        self.data = data
        self.test_data = test_data
        self.client_id = client_id
        self.random_state = random_state
        self.deterministic = deterministic

    def get_parameters(self, config: dict[str, Any] | None = None) -> list:
        return self.parameter_adapter.get_parameters(self.model)

    def set_parameters(self, params: list) -> None:
        self.parameter_adapter.set_parameters(self.model, params)

    def fit(
        self,
        params: list,
        config: dict[str, Any] | None = None,
    ) -> tuple[list, int, dict[str, Any]]:
        server_round, client_seed = self._seed_for_call(config, phase="fit")
        self.set_parameters(params)
        started_at = time.perf_counter()
        self._train(config)
        local_training_time = time.perf_counter() - started_at
        X_train, _ = self.data
        telemetry = {
            f"{FAL_TELEMETRY_PREFIX}client_id": self.client_id,
            f"{FAL_TELEMETRY_PREFIX}server_round": server_round,
            f"{FAL_TELEMETRY_PREFIX}local_training_time": local_training_time,
            f"{FAL_TELEMETRY_PREFIX}attack_generation_train_time": float(
                getattr(self.trainer, "last_attack_generation_train_time", 0.0)
            ),
        }
        if client_seed is not None:
            telemetry[f"{FAL_TELEMETRY_PREFIX}seed"] = client_seed
        return self.get_parameters(config), len(X_train), telemetry

    def evaluate(
        self,
        params: list,
        config: dict[str, Any] | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        server_round, client_seed = self._seed_for_call(config, phase="evaluate")
        self.set_parameters(params)
        data = self.test_data or self.data
        started_at = time.perf_counter()
        result = self._evaluate(data)
        evaluation_time = time.perf_counter() - started_at
        _, y = data
        telemetry = {
            f"{FAL_TELEMETRY_PREFIX}client_id": self.client_id,
            f"{FAL_TELEMETRY_PREFIX}server_round": server_round,
            f"{FAL_TELEMETRY_PREFIX}evaluation_time": evaluation_time,
            f"{FAL_TELEMETRY_PREFIX}attack_generation_evaluation_time": float(
                getattr(self.trainer, "last_attack_generation_evaluation_time", 0.0)
            ),
        }
        if client_seed is not None:
            telemetry[f"{FAL_TELEMETRY_PREFIX}seed"] = client_seed
        return self._to_flower_evaluation(result, len(y), telemetry=telemetry)

    def to_flower_client(self) -> Any:
        """Return a Flower ``Client`` object when Flower is installed."""

        outer = self

        class _NumPyClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                return outer.get_parameters(config)

            def fit(self, parameters, config):
                return outer.fit(parameters, config)

            def evaluate(self, parameters, config):
                return outer.evaluate(parameters, config)

        return _NumPyClient().to_client()

    def _train(self, fit_config: dict[str, Any] | None = None) -> None:
        if hasattr(self.trainer, "train"):
            self._train_with_config(fit_config)
            return

        X_train, y_train = self.data
        self.model.fit(X_train, y_train)

    def _train_with_config(self, fit_config: dict[str, Any] | None = None) -> None:
        if not fit_config:
            self.trainer.train(self.data)
            return

        trainer_params = getattr(self.trainer, "trainer_params", None)
        if not isinstance(trainer_params, dict):
            self.trainer.train(self.data)
            return

        original_params = dict(trainer_params)
        self.trainer.trainer_params = {**original_params, **fit_config}
        try:
            self.trainer.train(self.data)
        finally:
            self.trainer.trainer_params = original_params

    def _evaluate(self, data: tuple[Any, Any]) -> EvaluationResult:
        if hasattr(self.trainer, "evaluate"):
            result = self.trainer.evaluate(data)
            if isinstance(result, EvaluationResult):
                return result
            if isinstance(result, dict):
                loss = result.get("loss")
                metrics = {
                    name: value
                    for name, value in result.items()
                    if name != "loss"
                }
                return EvaluationResult(loss=loss, metrics=metrics)
            return EvaluationResult()

        X_test, y_test = data
        y_pred = self.model.predict(X_test)
        if hasattr(self.trainer, "metrics_manager"):
            return EvaluationResult(metrics=self.trainer.metrics_manager.compute_all(y_test, y_pred))
        return EvaluationResult()

    def _to_flower_evaluation(
        self,
        result: EvaluationResult,
        num_examples: int,
        telemetry: dict[str, Any] | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        # Flower requires a float loss. NaN is used only as an explicit missing
        # value and is removed before FALResult is constructed or exported.
        loss = math.nan if result.loss is None else float(result.loss)
        metrics = {
            name: encode_metric_value(value)
            for name, value in result.metrics.items()
        }
        metrics.update(telemetry or {})
        return loss, num_examples, metrics

    def _seed_for_call(
        self,
        config: dict[str, Any] | None,
        phase: str,
    ) -> tuple[int, int | None]:
        server_round = int((config or {}).get(FAL_SERVER_ROUND_KEY, 0))
        client_seed = derive_client_seed(
            self.random_state,
            client_id=self.client_id,
            server_round=server_round,
            phase=phase,
        )
        seed_everything(client_seed, deterministic=self.deterministic)
        return server_round, client_seed
