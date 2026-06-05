"""Concrete evaluation losses."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

try:
    import torch
except ModuleNotFoundError:
    torch = None

from fal.evaluation.losses.base import BaseEvaluationLoss


class LogLoss(BaseEvaluationLoss):
    name = "log_loss"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def compute(self, model: Any, X: Any, y_true: Any) -> float:
        raw_model = model.get_raw_model() if hasattr(model, "get_raw_model") else model
        if not callable(getattr(raw_model, "predict_proba", None)):
            raise ValueError("LogLoss requires a model with `predict_proba(X)`.")

        y_score = raw_model.predict_proba(X)
        return float(log_loss(y_true, y_score, **self.kwargs))


class MSELoss(BaseEvaluationLoss):
    name = "mse"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def compute(self, model: Any, X: Any, y_true: Any) -> float:
        y_pred = model.predict(X)
        return float(mean_squared_error(y_true, y_pred, **self.kwargs))


class ModelLoss(BaseEvaluationLoss):
    name = "model_loss"

    def compute(self, model: Any, X: Any, y_true: Any) -> float:
        criterion = getattr(model, "criterion", None)
        if criterion is None:
            raise ValueError("ModelLoss requires the model adapter to expose `criterion`.")

        if getattr(model, "framework", None) == "pytorch":
            return _compute_pytorch_model_loss(model, X, y_true, criterion)

        raise ValueError(f"ModelLoss is not supported for framework `{getattr(model, 'framework', None)}`.")


def _compute_pytorch_model_loss(model: Any, X: Any, y_true: Any, criterion: Any) -> float:
    if torch is None:
        raise ImportError("ModelLoss for PyTorch requires `torch`.")

    raw_model = model.get_raw_model()
    device = getattr(model, "device", None)

    X_tensor = torch.as_tensor(_writable_array(X))
    if not torch.is_floating_point(X_tensor):
        X_tensor = X_tensor.float()

    y_tensor = torch.as_tensor(_writable_array(y_true)).long()
    if device is not None:
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

    raw_model.eval()
    with torch.no_grad():
        outputs = raw_model(X_tensor)
        loss = criterion(outputs, y_tensor)

    return float(loss.detach().cpu().item())


def _writable_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return array
