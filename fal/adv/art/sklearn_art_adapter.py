"""Sklearn ART adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fal.adv.art.art_model_adapter import ARTModelAdapter


class SklearnARTAdapter(ARTModelAdapter):
    """Create an ART sklearn classifier."""

    def __init__(self) -> None:
        self.classifier_cls = _require_sklearn_classifier()

    def create_estimator(
        self,
        model: Any,
        config: dict[str, Any],
        X: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        try:
            return self.classifier_cls(
                model=model.get_raw_model(),
                clip_values=config.get("clip_values"),
            )
        except Exception as exc:
            raise ValueError(
                "This sklearn model is not supported by ART attacks yet. "
                "Tree-based models will be handled in a future version."
            ) from exc


def _require_sklearn_classifier() -> Any:
    try:
        classification = import_module("art.estimators.classification")
    except ImportError as exc:
        raise ImportError("Sklearn ART adapter requires `adversarial-robustness-toolbox`.") from exc
    return classification.SklearnClassifier
