"""Array dataset loader."""

from __future__ import annotations

from typing import Any

from fal.dataset.base import DatasetLoader


class ArrayDatasetLoader(DatasetLoader):
    """Return user-provided arrays as a dataset."""

    def __init__(self, X: Any, y: Any) -> None:
        self.X = X
        self.y = y

    def load(self) -> tuple[Any, Any]:
        if self.X is None or self.y is None:
            raise ValueError("ArrayDatasetLoader requires both `X` and `y`.")
        return self.X, self.y
