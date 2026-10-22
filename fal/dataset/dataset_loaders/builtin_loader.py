"""Built-in dataset loader."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from fal.dataset.base import DatasetLoader


class BuiltinDatasetLoader(DatasetLoader):
    """Load a named built-in dataset."""

    SUPPORTED_DATASETS: dict[
        str,
        Callable[["BuiltinDatasetLoader"], tuple[Any, Any]] | type[DatasetLoader],
    ] = {}

    @classmethod
    def register(cls, name: str, loader_cls: type[DatasetLoader]) -> None:
        """Register a named dataset loader without modifying FAL source code."""
        if not isinstance(loader_cls, type) or not issubclass(loader_cls, DatasetLoader):
            raise TypeError("Registered dataset loaders must inherit from DatasetLoader.")
        cls.SUPPORTED_DATASETS[_normalize_name(name)] = loader_cls

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls.SUPPORTED_DATASETS))

    def __init__(self, name: str, **params: Any) -> None:
        self.name = _normalize_name(name)
        self.params = params
        self.sklearn_datasets = None
        self.np = None
        self.torch = None
        self.torchvision_datasets = None
        self.transforms = None

        if self.name in {"digits", "iris", "wine"}:
            self.sklearn_datasets = _require_sklearn_datasets()
        elif self.name in {"cifar10", "fashion_mnist", "mnist"}:
            (
                self.np,
                self.torch,
                self.torchvision_datasets,
                self.transforms,
            ) = _require_torchvision_dataset_deps()

    def load(self) -> tuple[Any, Any]:
        loader = self.SUPPORTED_DATASETS.get(self.name)
        if loader is None:
            supported = ", ".join(sorted(self.SUPPORTED_DATASETS))
            raise ValueError(f"Unsupported built-in dataset `{self.name}`. Supported: {supported}.")

        if isinstance(loader, type) and issubclass(loader, DatasetLoader):
            return loader(**self.params).load()
        return loader(self)

    def _load_iris(self) -> tuple[Any, Any]:
        return self._load_sklearn_dataset("load_iris")

    def _load_digits(self) -> tuple[Any, Any]:
        return self._load_sklearn_dataset("load_digits")

    def _load_wine(self) -> tuple[Any, Any]:
        return self._load_sklearn_dataset("load_wine")

    def _load_mnist(self) -> tuple[Any, Any]:
        return self._load_torchvision_dataset("mnist")

    def _load_fashion_mnist(self) -> tuple[Any, Any]:
        return self._load_torchvision_dataset("fashion_mnist")

    def _load_cifar10(self) -> tuple[Any, Any]:
        return self._load_torchvision_dataset("cifar10")

    def _load_sklearn_dataset(self, loader_name: str) -> tuple[Any, Any]:
        bunch = getattr(self.sklearn_datasets, loader_name)()
        return self._apply_sample(bunch.data, bunch.target)

    def _load_torchvision_dataset(self, name: str) -> tuple[Any, Any]:
        root = self.params.get("root", "./data")
        train = bool(self.params.get("train", True))
        download = bool(self.params.get("download", True))
        transform = self.params.get("transform", self.transforms.ToTensor())

        dataset_cls = {
            "cifar10": self.torchvision_datasets.CIFAR10,
            "mnist": self.torchvision_datasets.MNIST,
            "fashion_mnist": self.torchvision_datasets.FashionMNIST,
        }[name]
        dataset = dataset_cls(root=root, train=train, download=download, transform=transform)
        X = self.torch.stack([sample for sample, _ in dataset]).numpy()
        if name in {"mnist", "fashion_mnist"}:
            X = X.reshape(len(X), -1)
        y = self.np.asarray([label for _, label in dataset])
        return self._apply_sample(X, y)

    def _apply_sample(self, X: Any, y: Any) -> tuple[Any, Any]:
        samples_per_class = self.params.get("samples_per_class")
        sample_size = self.params.get("sample_size")
        if samples_per_class is None and sample_size is None:
            return X, y

        np = self.np or _require_numpy()
        y_array = np.asarray(y)
        rng = np.random.default_rng(self.params.get("random_state"))

        if samples_per_class is not None:
            selected = []
            for label in np.unique(y_array):
                label_indices = np.flatnonzero(y_array == label)
                count = min(int(samples_per_class), len(label_indices))
                selected.extend(rng.choice(label_indices, size=count, replace=False))
            indices = np.asarray(selected)
        else:
            count = min(int(sample_size), len(y_array))
            indices = rng.choice(len(y_array), size=count, replace=False)

        rng.shuffle(indices)
        return X[indices], y_array[indices]


BuiltinDatasetLoader.SUPPORTED_DATASETS = {
    "cifar10": BuiltinDatasetLoader._load_cifar10,
    "digits": BuiltinDatasetLoader._load_digits,
    "fashion_mnist": BuiltinDatasetLoader._load_fashion_mnist,
    "iris": BuiltinDatasetLoader._load_iris,
    "mnist": BuiltinDatasetLoader._load_mnist,
    "wine": BuiltinDatasetLoader._load_wine,
}


def _normalize_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Dataset name must be a non-empty string.")
    return value.strip().lower().replace("-", "_")


def _require_sklearn_datasets() -> Any:
    try:
        datasets = import_module("sklearn.datasets")
    except ImportError as exc:
        raise ImportError("Built-in sklearn datasets require `scikit-learn`.") from exc
    return datasets


def _require_numpy() -> Any:
    try:
        np = import_module("numpy")
    except ImportError as exc:
        raise ImportError("Dataset sampling requires `numpy`.") from exc
    return np


def _require_torchvision_dataset_deps() -> tuple[Any, Any, Any, Any]:
    try:
        np = import_module("numpy")
        torch = import_module("torch")
        datasets = import_module("torchvision.datasets")
        transforms = import_module("torchvision.transforms")
    except ImportError as exc:
        raise ImportError(
            "Torchvision datasets require `numpy`, `torch`, and `torchvision`."
        ) from exc
    return np, torch, datasets, transforms
