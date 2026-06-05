"""Adapter for scikit-learn compatible estimators."""

from __future__ import annotations

from inspect import signature
from typing import Any
import warnings

from sklearn.exceptions import ConvergenceWarning

from fal.models.base import BaseModel


class SklearnModelAdapter(BaseModel):
    """Wrap a scikit-learn style estimator behind the ``BaseModel`` interface."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._validate_model()

    @property
    def framework(self) -> str:
        return "sklearn"

    def fit(self, X: Any, y: Any, **params: Any) -> "SklearnModelAdapter":
        if float(params.get("proximal_mu", 0.0) or 0.0) > 0:
            raise NotImplementedError(
                "FedProx proximal training is not supported for sklearn models. "
                "Use `fl_algorithm='fedavg'` or a PyTorch model for FedProx."
            )

        if callable(getattr(self.model, "partial_fit", None)):
            partial_fit_kwargs = self._partial_fit_kwargs(params)
            self.model.partial_fit(X, y, **partial_fit_kwargs)
            return self

        self._prepare_warm_start_model(params)
        if params.get("suppress_convergence_warnings", False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.model.fit(X, y)
            return self

        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def get_raw_model(self) -> Any:
        return self.model

    def _validate_model(self) -> None:
        if not callable(getattr(self.model, "fit", None)):
            raise TypeError("A sklearn model must expose a callable `fit(X, y)` method.")
        if not callable(getattr(self.model, "predict", None)):
            raise TypeError("A sklearn model must expose a callable `predict(X)` method.")

    def _partial_fit_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        classes = params.get("classes")
        if classes is None:
            return {}

        try:
            parameters = signature(self.model.partial_fit).parameters
        except (TypeError, ValueError):
            return {}

        if "classes" not in parameters:
            return {}

        if hasattr(self.model, "classes_"):
            return {}

        return {"classes": classes}

    def _prepare_warm_start_model(self, params: dict[str, Any]) -> None:
        if not hasattr(self.model, "warm_start"):
            return

        if not params.get("auto_warm_start", True):
            return

        self.model.warm_start = True

        local_max_iter = params.get("local_max_iter")
        if local_max_iter is not None and hasattr(self.model, "max_iter"):
            self.model.max_iter = int(local_max_iter)
