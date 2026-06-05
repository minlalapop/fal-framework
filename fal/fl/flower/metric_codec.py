"""Encode non-scalar FAL metrics for Flower transport."""

from __future__ import annotations

import json
from typing import Any


FAL_JSON_METRIC_PREFIX = "__fal_json_metric__:"
FLOWER_SCALAR_TYPES = (bool, bytes, float, int, str)


def encode_metric_value(value: Any) -> Any:
    """Return a Flower-compatible metric value."""

    if isinstance(value, FLOWER_SCALAR_TYPES):
        return value

    return FAL_JSON_METRIC_PREFIX + json.dumps(value, default=_to_jsonable)


def decode_metric_value(value: Any) -> Any:
    """Restore a FAL metric value encoded for Flower."""

    if isinstance(value, str) and value.startswith(FAL_JSON_METRIC_PREFIX):
        return json.loads(value[len(FAL_JSON_METRIC_PREFIX):])
    return value


def decode_metric_history(
    metrics: dict[str, list[tuple[int, Any]]],
) -> dict[str, list[tuple[int, Any]]]:
    """Decode Flower metric history into framework-level values."""

    return {
        name: [(round_number, decode_metric_value(value)) for round_number, value in values]
        for name, values in metrics.items()
    }


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(
        f"Metric value of type {type(value).__name__} is not JSON serializable."
    )
