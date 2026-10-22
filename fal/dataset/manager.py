"""Dataset preparation manager."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fal.dataset.base import DatasetLoader
from fal.dataset.client_splitters.client_splitter import ClientSplitter
from fal.dataset.client_splitters.iid_splitter import IIDSplitter
from fal.dataset.client_splitters.splitter_utils import shuffle as shuffle_data, take


class DatasetManager:
    """Load a dataset, hold out a test set, and split train data into clients."""

    def __init__(
        self,
        loader: DatasetLoader,
        splitter: ClientSplitter | None = None,
        num_clients: int = 2,
        test_size: float = 0.2,
        random_state: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.loader = loader
        self.splitter = splitter or IIDSplitter(shuffle_data=shuffle, random_state=random_state)
        self.num_clients = num_clients
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.train_test_split = _optional_train_test_split()
        self.clients_data: list[tuple[Any, Any]] | None = None
        self.test_data: tuple[Any, Any] | None = None

    def prepare(self) -> "DatasetManager":
        loaded = self.loader.load()
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
            raise TypeError("DatasetLoader.load() must return exactly `(X, y)`.")
        X, y = loaded
        try:
            num_samples = len(X)
            num_labels = len(y)
        except TypeError as exc:
            raise TypeError("Dataset features and labels must be sized collections.") from exc
        if num_samples != num_labels:
            raise ValueError(
                "Dataset features and labels must contain the same number of samples."
            )
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)

        self.clients_data = self.splitter.split(X_train, y_train, self.num_clients)
        self.test_data = (X_test, y_test)
        return self

    def get_clients_data(self) -> list[tuple[Any, Any]]:
        if self.clients_data is None:
            raise RuntimeError("DatasetManager.prepare() must be called before get_clients_data().")
        return self.clients_data

    def get_test_data(self) -> tuple[Any, Any]:
        if self.test_data is None:
            raise RuntimeError("DatasetManager.prepare() must be called before get_test_data().")
        return self.test_data

    def _train_test_split(self, X: Any, y: Any) -> tuple[Any, Any, Any, Any]:
        self._validate_split_params()

        if self.train_test_split is None:
            return self._fallback_train_test_split(X, y)

        return self.train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

    def _validate_split_params(self) -> None:
        if not isinstance(self.num_clients, int) or self.num_clients <= 0:
            raise ValueError("`num_clients` must be a positive integer.")
        if not 0 < self.test_size < 1:
            raise ValueError("`test_size` must be between 0 and 1.")

    def _fallback_train_test_split(self, X: Any, y: Any) -> tuple[Any, Any, Any, Any]:
        n_samples = len(y)
        indices = list(range(n_samples))
        if self.shuffle:
            shuffle_data(indices, self.random_state)

        n_test = max(1, int(round(n_samples * self.test_size)))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        return (
            take(X, train_indices),
            take(X, test_indices),
            take(y, train_indices),
            take(y, test_indices),
        )


def _optional_train_test_split() -> Any | None:
    try:
        model_selection = import_module("sklearn.model_selection")
    except ImportError:
        return None
    return model_selection.train_test_split
