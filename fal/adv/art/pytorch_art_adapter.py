"""PyTorch ART adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from fal.adv.art.art_model_adapter import ARTModelAdapter


class PyTorchARTAdapter(ARTModelAdapter):
    """Create an ART PyTorch classifier."""

    def __init__(self) -> None:
        self.classifier_cls = _require_pytorch_classifier()

    def create_estimator(
        self,
        model: Any,
        config: dict[str, Any],
        X: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        config = self._complete_config(model, config, X, y)
        criterion = config.get("criterion") or getattr(model, "criterion", None)
        optimizer = config.get("optimizer") or getattr(model, "optimizer", None)

        if criterion is None or optimizer is None:
            raise ValueError("PyTorchARTAdapter requires `criterion` and `optimizer`.")

        return self.classifier_cls(
            model=model.get_raw_model(),
            loss=criterion,
            optimizer=optimizer,
            input_shape=config["input_shape"],
            nb_classes=config["nb_classes"],
            clip_values=config.get("clip_values"),
        )

    def _complete_config(
        self,
        model: Any,
        config: dict[str, Any],
        X: Any | None,
        y: Any | None,
    ) -> dict[str, Any]:
        completed = dict(config)

        if completed.get("input_shape") is None:
            completed["input_shape"] = _infer_input_shape(X)

        if completed.get("nb_classes") is None:
            completed["nb_classes"] = _infer_nb_classes(model, X, y)

        if completed.get("clip_values") is None:
            completed["clip_values"] = _infer_clip_values(X)

        return completed


def _require_pytorch_classifier() -> Any:
    try:
        classification = import_module("art.estimators.classification")
    except ImportError as exc:
        raise ImportError("PyTorch ART adapter requires `adversarial-robustness-toolbox`.") from exc
    return classification.PyTorchClassifier


def _infer_input_shape(X: Any | None) -> tuple[int, ...]:
    if X is None or not hasattr(X, "shape") or len(X.shape) < 2:
        raise ValueError("PyTorchARTAdapter could not infer `input_shape` from X.")
    return tuple(X.shape[1:])


def _infer_nb_classes(model: Any, X: Any | None, y: Any | None) -> int:
    nb_classes = _infer_nb_classes_from_model(model, X)
    if nb_classes is not None:
        return nb_classes

    if y is None:
        raise ValueError("PyTorchARTAdapter could not infer `nb_classes` from model output or y.")
    return int(len(np.unique(np.asarray(y))))


def _infer_nb_classes_from_model(model: Any, X: Any | None) -> int | None:
    if X is None:
        return None

    torch = getattr(model, "_torch", None)
    if torch is None:
        return None

    raw_model = model.get_raw_model()
    device = getattr(model, "device", None)
    sample = np.asarray(X[:1]).copy()
    tensor = torch.as_tensor(sample)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)

    raw_model.eval()
    with torch.no_grad():
        output = raw_model(tensor)

    if getattr(output, "ndim", 0) > 1:
        return int(output.shape[-1])
    return 1


def _infer_clip_values(X: Any | None) -> tuple[float, float] | None:
    if X is None:
        return None

    values = np.asarray(X)
    if values.size == 0:
        return None

    min_value = float(values.min())
    max_value = float(values.max())

    if min_value >= 0.0 and max_value <= 1.0:
        return 0.0, 1.0

    return min_value, max_value
