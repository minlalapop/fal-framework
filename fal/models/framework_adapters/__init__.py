"""Model framework adapters."""

from fal.models.framework_adapters.pytorch_adapter import PyTorchModelAdapter
from fal.models.framework_adapters.sklearn_adapter import SklearnModelAdapter

__all__ = [
    "PyTorchModelAdapter",
    "SklearnModelAdapter",
]
