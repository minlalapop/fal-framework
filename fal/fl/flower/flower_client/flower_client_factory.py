"""Factory for Flower clients."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from fal.fl.flower.flower_client.flower_client_adapter import FlowerClientAdapter
from fal.fl.flower.flower_client.flower_parameters.flower_parameter_factory import (
    ParameterAdapterFactory,
)


class FlowerClientFactory:
    """Create one independent Flower client per data partition."""

    _registry = {
        "flower_client": FlowerClientAdapter,
    }

    @classmethod
    def register(cls, name: str, client_cls: type[FlowerClientAdapter]) -> None:
        cls._registry[cls._normalize_name(name)] = client_cls

    def create_clients(
        self,
        model: Any,
        trainer: Any,
        clients_data: list[tuple[Any, Any]],
        test_data: tuple[Any, Any] | None = None,
    ) -> list[FlowerClientAdapter]:
        client_cls = self._get_client_class()
        classes = self._get_global_classes(clients_data)
        return [
            client_cls(
                model=client_model,
                trainer=self._clone_trainer(trainer, client_model, classes),
                parameter_adapter=self._create_parameter_adapter(client_model, data, classes),
                data=data,
                test_data=test_data,
            )
            for client_model, data in (
                (self._clone_model(model), client_data)
                for client_data in clients_data
            )
        ]

    def _clone_model(self, model: Any) -> Any:
        if callable(getattr(model, "clone", None)):
            return model.clone()

        try:
            return copy.deepcopy(model)
        except Exception as exc:
            raise ValueError(
                "Each FL client requires an independent model instance, "
                "but this model could not be cloned."
            ) from exc

    def _clone_trainer(self, trainer: Any, model: Any, classes: Any) -> Any:
        trainer_params = dict(getattr(trainer, "trainer_params", {}))
        trainer_params.setdefault("classes", classes)

        return type(trainer)(
            model=model,
            attack=getattr(trainer, "attack", None),
            metrics_manager=getattr(trainer, "metrics_manager", None),
            loss_manager=getattr(trainer, "loss_manager", None),
            trainer_params=trainer_params,
        )

    def _create_parameter_adapter(self, model: Any, data: tuple[Any, Any], classes: Any) -> Any:
        return ParameterAdapterFactory.create(model, data=data, classes=classes)

    def _get_global_classes(self, clients_data: list[tuple[Any, Any]]) -> Any:
        labels = [np.asarray(y) for _, y in clients_data]
        return np.unique(np.concatenate(labels))

    @classmethod
    def _get_client_class(cls) -> type[FlowerClientAdapter]:
        return cls._registry["flower_client"]

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Flower client key must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
