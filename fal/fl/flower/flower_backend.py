"""Flower implementation of the FL backend interface."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fal.fl.backend_base import FLBackend
from fal.fl.flower.flower_client.flower_client_factory import FlowerClientFactory
from fal.fl.flower.flower_client.flower_parameters.flower_parameter_factory import (
    ParameterAdapterFactory,
)
from fal.fl.flower.metric_codec import decode_metric_history
from fal.fl.flower.flower_server.flower_server_adapter import FlowerServerAdapter
from fal.result import FALResult
from fal.trainers.mixed import MixedTrainer


class FlowerBackend(FLBackend):
    """Flower-specific FL backend orchestration."""

    def __init__(
        self,
        client_factory: FlowerClientFactory | None = None,
        server: FlowerServerAdapter | None = None,
    ) -> None:
        self.client_factory = client_factory or FlowerClientFactory()
        self.server = server or FlowerServerAdapter()

    def start(
        self,
        model: Any,
        trainer: Any,
        clients_data: list,
        test_data: Any,
        strategy: Any,
        num_rounds: int = 1,
        verbose: int = 1,
        run_config: dict[str, Any] | None = None,
    ) -> Any:
        run_config = run_config or {}
        clients = self.client_factory.create_clients(
            model=model,
            trainer=trainer,
            clients_data=clients_data,
            test_data=test_data,
            random_state=run_config.get("random_state"),
            deterministic=bool(run_config.get("deterministic", False)),
        )

        server_result = self.server.start_server(
            clients=clients,
            strategy=strategy,
            num_rounds=num_rounds,
            verbose=verbose,
            run_config=run_config,
        )

        self._apply_final_parameters(
            model,
            server_result.final_parameters,
            clients_data,
            trainer=trainer,
        )
        history = server_result.history
        metrics = decode_metric_history(dict(getattr(history, "metrics_distributed", {})))

        return FALResult(
            model=model,
            history=history,
            losses=_available_losses(getattr(history, "losses_distributed", [])),
            metrics=metrics,
            backend="flower",
            raw_result=history,
            timings=server_result.timings,
        )

    def _apply_final_parameters(
        self,
        model: Any,
        final_parameters: list[Any] | None,
        clients_data: list,
        trainer: Any = None,
    ) -> None:
        if final_parameters is None:
            return

        data = clients_data[0] if clients_data else None
        classes = self._get_global_classes(clients_data)
        parameter_adapter = ParameterAdapterFactory.create(model, data=data, classes=classes)
        if isinstance(trainer, MixedTrainer):
            parameter_count = len(parameter_adapter.get_parameters(model))
            final_parameters = final_parameters[parameter_count:]
        parameter_adapter.set_parameters(model, final_parameters)
        configure_device = getattr(model, "configure_runtime_device", None)
        if callable(configure_device):
            configure_device()

    def _get_global_classes(self, clients_data: list[tuple[Any, Any]]) -> Any:
        if not clients_data:
            return None

        labels = [np.asarray(y) for _, y in clients_data]
        return np.unique(np.concatenate(labels))


def _available_losses(losses: Any) -> list[tuple[int, float]]:
    """Remove Flower's NaN marker for evaluations without a configured loss."""
    return [
        (int(round_number), float(loss))
        for round_number, loss in losses
        if not math.isnan(float(loss))
    ]
