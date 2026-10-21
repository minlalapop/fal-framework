"""Factory for Flower clients."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from fal.fl.flower.flower_client.flower_client_adapter import FlowerClientAdapter
from fal.fl.flower.flower_client.flower_parameters.flower_parameter_factory import (
    ParameterAdapterFactory,
)
from fal.trainers.mixed import MixedTrainer


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
        random_state: int | None = None,
        deterministic: bool = False,
    ) -> list[FlowerClientAdapter]:
        client_cls = self._get_client_class()
        classes = self._get_global_classes(clients_data)
        clients = []
        for client_id, data in enumerate(clients_data):
            client_model = self._clone_model(model)
            secondary_model = self._clone_model(model) if isinstance(trainer, MixedTrainer) else None
            clients.append(
                self._create_client(
                    client_cls=client_cls,
                    model=client_model,
                    secondary_model=secondary_model,
                    trainer=trainer,
                    data=data,
                    test_data=test_data,
                    classes=classes,
                    client_id=client_id,
                    random_state=random_state,
                    deterministic=deterministic,
                )
            )
        return clients

    def _create_client(
        self,
        client_cls: type[FlowerClientAdapter],
        model: Any,
        secondary_model: Any,
        trainer: Any,
        data: tuple[Any, Any],
        test_data: tuple[Any, Any] | None,
        classes: Any,
        client_id: int,
        random_state: int | None,
        deterministic: bool,
    ) -> FlowerClientAdapter:
        # Construct with the historical signature so registered custom client
        # adapters remain compatible, then attach the optional runtime controls.
        parameter_adapter = self._create_parameter_adapter(model, data, classes)
        secondary_parameter_adapter = (
            self._create_parameter_adapter(secondary_model, data, classes)
            if secondary_model is not None
            else None
        )
        client = client_cls(
            model=model,
            trainer=self._clone_trainer(trainer, model, classes, secondary_model),
            parameter_adapter=parameter_adapter,
            data=data,
            test_data=test_data,
        )
        if secondary_model is not None:
            try:
                client.secondary_model = secondary_model
                client.secondary_parameter_adapter = secondary_parameter_adapter
            except AttributeError as exc:
                raise TypeError(
                    "A custom Flower client adapter must allow the secondary model "
                    "attributes required by MixedTrainer."
                ) from exc
        for name, value in (
            ("client_id", client_id),
            ("random_state", random_state),
            ("deterministic", deterministic),
        ):
            try:
                setattr(client, name, value)
            except AttributeError:
                pass
        return client

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

    def _clone_trainer(
        self,
        trainer: Any,
        model: Any,
        classes: Any,
        secondary_model: Any = None,
    ) -> Any:
        trainer_params = dict(getattr(trainer, "trainer_params", {}))
        trainer_params.setdefault("classes", classes)

        trainer_args = dict(
            model=model,
            attack=getattr(trainer, "attack", None),
            metrics_manager=getattr(trainer, "metrics_manager", None),
            loss_manager=getattr(trainer, "loss_manager", None),
            trainer_params=trainer_params,
        )
        if isinstance(trainer, MixedTrainer):
            trainer_args["adversarial_model"] = secondary_model
        return type(trainer)(**trainer_args)

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
