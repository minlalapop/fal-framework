"""ART model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ARTModelAdapter(ABC):
    """Interface for converting a model adapter into an ART estimator."""

    @abstractmethod
    def create_estimator(
        self,
        model: Any,
        config: dict[str, Any],
        X: Any | None = None,
        y: Any | None = None,
    ) -> Any:
        """Create an ART estimator."""


def validate_clip_values(
    clip_values: Any,
    X: Any | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray] | None:
    """Validate ART input bounds and their compatibility with the input shape."""
    if clip_values is None:
        return None
    if not isinstance(clip_values, (tuple, list)) or len(clip_values) != 2:
        raise ValueError("ART `clip_values` must be a (minimum, maximum) pair.")

    try:
        lower = np.asarray(clip_values[0], dtype=float)
        upper = np.asarray(clip_values[1], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ART `clip_values` bounds must be numeric.") from exc

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("ART `clip_values` bounds must be finite.")

    values = None if X is None else np.asarray(X)
    feature_shape = None
    if values is not None and values.ndim >= 2:
        feature_shape = values.shape[1:]

    if feature_shape is not None:
        try:
            lower_for_input = np.broadcast_to(lower, feature_shape)
            upper_for_input = np.broadcast_to(upper, feature_shape)
        except ValueError as exc:
            raise ValueError(
                "ART `clip_values` bounds are not broadcastable to input shape "
                f"{feature_shape}."
            ) from exc
    else:
        try:
            lower_for_input, upper_for_input = np.broadcast_arrays(lower, upper)
        except ValueError as exc:
            raise ValueError("ART `clip_values` minimum and maximum are incompatible.") from exc

    if not np.all(lower_for_input < upper_for_input):
        raise ValueError("ART `clip_values` minimum must be strictly below maximum.")

    if values is not None and values.size:
        if np.any(values < lower_for_input) or np.any(values > upper_for_input):
            raise ValueError("Input data contains values outside ART `clip_values`.")

    return _scalar_or_array(lower), _scalar_or_array(upper)


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value
