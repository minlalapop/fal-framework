"""CSV dataset loader."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from fal.dataset.base import DatasetLoader


class CustomCSVDatasetLoader(DatasetLoader):
    """Load a supervised dataset from a CSV file."""

    def __init__(self, path: str | Path, target: str) -> None:
        self.pd = _require_pandas()
        self.path = Path(path)
        self.target = target

    def load(self) -> tuple[Any, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV dataset not found: {self.path}")

        dataframe = self.pd.read_csv(self.path)
        if self.target not in dataframe.columns:
            raise ValueError(f"Target column `{self.target}` not found in {self.path}.")

        X = dataframe.drop(columns=[self.target])
        y = dataframe[self.target]
        return X, y


def _require_pandas() -> Any:
    try:
        pd = import_module("pandas")
    except ImportError as exc:
        raise ImportError("CSV dataset loading requires `pandas`.") from exc
    return pd
