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
    _detectors: dict[str, Any] = {}

    @classmethod
    def register(
        cls,
        name: str,
        adapter_cls: type[BaseModel],
        detector: Any | None = None,
    ) -> None:
        name = cls._normalize_name(name)
        if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, BaseModel):
            raise TypeError("Registered model adapters must inherit from BaseModel.")

        if name not in {"pytorch", "sklearn"}:
            detector = detector or getattr(adapter_cls, "supports", None)
            if not callable(detector):
                raise TypeError(
                    "A custom model adapter requires a callable `detector` or "
                    "a callable `supports(model)` method."
                )
            cls._detectors[name] = detector

        cls._registry[name] = adapter_cls

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

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

        adapter = adapter_cls(model, **params)
        if not isinstance(adapter, BaseModel):
            raise TypeError("A registered model adapter must create a BaseModel instance.")
        return adapter

    @classmethod
    def _detect_model_type(cls, model: Any) -> str:
        matches = []
        if _is_pytorch_model(model):
            matches.append("pytorch")
        elif _is_sklearn_instance(model):
            matches.append("sklearn")

        for name, detector in cls._detectors.items():
            try:
                if detector(model):
                    matches.append(name)
            except Exception as exc:
                raise TypeError(f"Model detector for `{name}` failed.") from exc

        if len(matches) > 1:
            names = ", ".join(sorted(matches))
            raise TypeError(
                f"Ambiguous model adapter detection: {names}. "
                "Pass an explicitly constructed BaseModel adapter as `model`."
            )
        if matches:
            return matches[0]
        if _looks_like_sklearn_estimator(model):
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
    return _is_sklearn_instance(model) or _looks_like_sklearn_estimator(model)


def _is_sklearn_instance(model: Any) -> bool:
    if BaseEstimator is None:
        return False
    return isinstance(model, BaseEstimator)


def _looks_like_sklearn_estimator(model: Any) -> bool:
    return callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None))
