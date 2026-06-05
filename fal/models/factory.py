"""Factory that turns user models into FAL model adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fal.models.base import BaseModel
from fal.models.framework_adapters.pytorch_adapter import PyTorchModelAdapter
from fal.models.framework_adapters.sklearn_adapter import SklearnModelAdapter

try:
    torch = import_module("torch")
except ImportError:
    torch = None

try:
    BaseEstimator = import_module("sklearn.base").BaseEstimator
except ImportError:
    BaseEstimator = None


class ModelAdapterFactory:
    """Create the correct ``BaseModel`` adapter for a user-provided model."""

    _registry = {
        "pytorch": PyTorchModelAdapter,
        "sklearn": SklearnModelAdapter,
    }

    @classmethod
    def register(cls, name: str, adapter_cls: type[BaseModel]) -> None:
        cls._registry[cls._normalize_name(name)] = adapter_cls

    @classmethod
    def create(cls, model: Any, model_params: dict[str, Any] | None = None) -> BaseModel:
        if isinstance(model, BaseModel):
            return model

        params = model_params or {}
        model_type = cls._detect_model_type(model)
        adapter_cls = cls._registry.get(model_type)

        if adapter_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise TypeError(f"Unsupported model type `{model_type}`. Supported: {supported}.")

        if model_type == "sklearn":
            return adapter_cls(model)

        return adapter_cls(model, **params)

    @staticmethod
    def _detect_model_type(model: Any) -> str:
        if _is_pytorch_model(model):
            return "pytorch"
        if _is_sklearn_model(model):
            return "sklearn"

        raise TypeError(
            "Unsupported model type. Provide a scikit-learn estimator, "
            "a torch.nn.Module, or an object already implementing BaseModel."
        )

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model adapter key must be a non-empty string.")
        return value.strip().lower().replace("-", "_")


def _is_pytorch_model(model: Any) -> bool:
    if torch is None:
        return False

    return isinstance(model, torch.nn.Module)


def _is_sklearn_model(model: Any) -> bool:
    if BaseEstimator is None:
        return _looks_like_sklearn_estimator(model)

    return isinstance(model, BaseEstimator) or _looks_like_sklearn_estimator(model)


def _looks_like_sklearn_estimator(model: Any) -> bool:
    return callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None))
