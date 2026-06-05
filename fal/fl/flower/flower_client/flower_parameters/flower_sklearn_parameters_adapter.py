"""Sklearn Flower parameter adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

from fal.fl.flower.flower_client.flower_parameters.flower_parameters_adapter import (
    FlowerParameterAdapter,
)


class SklearnParameterAdapter(FlowerParameterAdapter):
    SUPPORTED_UNFITTED_MODELS = {
        "LogisticRegression",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "SGDClassifier",
        "LinearRegression",
        "SGDRegressor",
    }

    def supports(self, model: Any) -> bool:
        if getattr(model, "framework", None) != "sklearn":
            return False
        raw = model.get_raw_model()
        return (
            _has_custom_parameter_hooks(raw)
            or _has_linear_parameters(raw)
            or type(raw).__name__ in self.SUPPORTED_UNFITTED_MODELS
        )

    def initialize(self, model: Any, X: Any, y: Any, classes: Any | None = None) -> None:
        raw = model.get_raw_model()
        if _has_custom_parameter_hooks(raw):
            initializer = getattr(raw, "initialize_fal_parameters", None)
            if callable(initializer):
                initializer(X, y, classes=classes)
            return

        if _has_linear_parameters(raw):
            return

        if type(raw).__name__ not in self.SUPPORTED_UNFITTED_MODELS:
            raise ValueError(
                _unsupported_model_message(raw)
            )

        n_features = _num_features(X)

        if isinstance(raw, ClassifierMixin):
            classes = np.asarray(classes if classes is not None else np.unique(y))
            n_classes = len(classes)
            coef_rows = 1 if n_classes == 2 else n_classes

            raw.classes_ = classes
            raw.coef_ = np.zeros((coef_rows, n_features), dtype=float)
            raw.intercept_ = np.zeros(coef_rows, dtype=float)
            raw.n_features_in_ = n_features
            return

        y_array = np.asarray(y)
        if y_array.ndim > 1 and y_array.shape[1] > 1:
            n_outputs = y_array.shape[1]
            raw.coef_ = np.zeros((n_outputs, n_features), dtype=float)
            raw.intercept_ = np.zeros(n_outputs, dtype=float)
        else:
            raw.coef_ = np.zeros(n_features, dtype=float)
            raw.intercept_ = 0.0

        raw.n_features_in_ = n_features

    def get_parameters(self, model: Any) -> list:
        raw = model.get_raw_model()

        if not self.supports(model):
            raise ValueError(
                _unsupported_model_message(raw)
            )

        if _has_custom_parameter_hooks(raw):
            return [np.asarray(value) for value in raw.get_fal_parameters()]

        return [np.asarray(raw.coef_), np.asarray(raw.intercept_)]

    def set_parameters(self, model: Any, parameters: list) -> None:
        raw = model.get_raw_model()
        if _has_custom_parameter_hooks(raw):
            raw.set_fal_parameters([np.asarray(value) for value in parameters])
            return

        raw.coef_ = np.asarray(parameters[0])
        raw.intercept_ = np.asarray(parameters[1])


def _num_features(X: Any) -> int:
    if hasattr(X, "shape") and len(X.shape) > 1:
        return X.shape[1]
    if hasattr(X, "iloc"):
        return X.shape[1]
    return len(X[0])


def _has_linear_parameters(model: Any) -> bool:
    return hasattr(model, "coef_") and hasattr(model, "intercept_")


def _has_custom_parameter_hooks(model: Any) -> bool:
    return callable(getattr(model, "get_fal_parameters", None)) and callable(
        getattr(model, "set_fal_parameters", None)
    )


def _unsupported_model_message(model: Any) -> str:
    return (
        f"{type(model).__name__} is not supported for federated parameter aggregation. "
        "Sklearn aggregation currently supports linear estimators with `coef_`/`intercept_`, "
        "a small set of unfitted linear estimators, or custom sklearn-compatible models "
        "that implement `get_fal_parameters()` and `set_fal_parameters(parameters)`."
    )
