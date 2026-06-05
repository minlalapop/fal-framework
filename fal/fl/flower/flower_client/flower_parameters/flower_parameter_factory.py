"""Factory for Flower parameter adapters."""

from __future__ import annotations

from typing import Any

from fal.fl.flower.flower_client.flower_parameters.flower_parameters_adapter import (
    FlowerParameterAdapter,
)
from fal.fl.flower.flower_client.flower_parameters.flower_sklearn_parameters_adapter import (
    SklearnParameterAdapter,
)

try:
    from fal.fl.flower.flower_client.flower_parameters.flower_pytorch_parameters_adapter import (
        PyTorchParameterAdapter,
    )
except ModuleNotFoundError:
    PyTorchParameterAdapter = None


class ParameterAdapterFactory:
    """Create Flower parameter adapters from model framework names."""

    _registry = {"sklearn": SklearnParameterAdapter}
    if PyTorchParameterAdapter is not None:
        _registry["pytorch"] = PyTorchParameterAdapter

    @classmethod
    def register(cls, name: str, adapter_cls: type[FlowerParameterAdapter]) -> None:
        cls._registry[cls._normalize_name(name)] = adapter_cls

    @classmethod
    def create(
        cls,
        model: Any,
        data: tuple[Any, Any] | None = None,
        classes: Any | None = None,
    ) -> FlowerParameterAdapter:
        framework = cls._normalize_name(getattr(model, "framework", type(model).__name__))
        adapter_cls = cls._registry.get(framework)

        if adapter_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"No Flower parameter adapter supports model framework `{framework}`. "
                f"Supported: {supported}."
            )

        adapter = adapter_cls()
        if not adapter.supports(model):
            raise ValueError(f"No Flower parameter adapter supports model framework `{framework}`.")

        if data is not None and hasattr(adapter, "initialize"):
            X, y = data
            adapter.initialize(model, X, y, classes=classes)

        return adapter

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Parameter adapter key must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
