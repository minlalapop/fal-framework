"""Shared helpers for dataset splitters."""

from __future__ import annotations

from importlib import import_module
import random
from typing import Any

try:
    np = import_module("numpy")
except ImportError:
    np = None


def validate_num_clients(num_clients: int) -> None:
    if not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError("`num_clients` must be a positive integer.")


def num_samples(values: Any) -> int:
    try:
        return len(values)
    except TypeError as exc:
        raise ValueError("Dataset values must be sized.") from exc


def split_indices(indices: list[int], num_parts: int) -> list[list[int]]:
    return [indices[index::num_parts] for index in range(num_parts)]


def split_contiguous_indices(indices: list[int], num_parts: int) -> list[list[int]]:
    """Split ordered indices into balanced, contiguous parts."""
    part_size, remainder = divmod(len(indices), num_parts)
    parts = []
    start = 0

    for part_index in range(num_parts):
        stop = start + part_size + (1 if part_index < remainder else 0)
        parts.append(indices[start:stop])
        start = stop

    return parts


def shuffle(values: list[Any], random_state: int | None) -> None:
    rng = random.Random(random_state)
    rng.shuffle(values)


def item(values: Any, index: int) -> Any:
    value = values[index]
    if hasattr(value, "item"):
        return value.item()
    return value


def take(values: Any, indices: list[int]) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[indices]

    if np is None:
        return [values[index] for index in indices]

    if isinstance(values, np.ndarray):
        return values[indices]

    try:
        return values[indices]
    except (TypeError, IndexError):
        return [values[index] for index in indices]
