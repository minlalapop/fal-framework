"""Built-in dataset loader."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from fal.dataset.base import DatasetLoader


class BuiltinDatasetLoader(DatasetLoader):
    """Load a named built-in dataset."""

    SUPPORTED_DATASETS: dict[str, Callable[["BuiltinDatasetLoader"], tuple[Any, Any]]] = {}

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
        elif self.name in {"fashion_mnist", "mnist"}:
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

    def _load_sklearn_dataset(self, loader_name: str) -> tuple[Any, Any]:
        bunch = getattr(self.sklearn_datasets, loader_name)()
        return bunch.data, bunch.target

    def _load_torchvision_dataset(self, name: str) -> tuple[Any, Any]:
        root = self.params.get("root", "./data")
        train = bool(self.params.get("train", True))
        download = bool(self.params.get("download", True))
        transform = self.params.get("transform", self.transforms.ToTensor())

        dataset_cls = {
            "mnist": self.torchvision_datasets.MNIST,
            "fashion_mnist": self.torchvision_datasets.FashionMNIST,
        }[name]
        dataset = dataset_cls(root=root, train=train, download=download, transform=transform)
        X = self.torch.stack([sample for sample, _ in dataset]).numpy()
        y = self.np.asarray([label for _, label in dataset])
        return X, y


BuiltinDatasetLoader.SUPPORTED_DATASETS = {
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


def _require_torchvision_dataset_deps() -> tuple[Any, Any, Any, Any]:
    try:
        np = import_module("numpy")
        torch = import_module("torch")
        datasets = import_module("torchvision.datasets")
        transforms = import_module("torchvision.transforms")
    except ImportError as exc:
        raise ImportError(
            "MNIST and Fashion-MNIST loading require `numpy`, `torch`, and `torchvision`."
        ) from exc
    return np, torch, datasets, transforms
