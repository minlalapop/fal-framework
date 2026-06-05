"""Model adapters exposed by the FAL framework."""

from fal.models.base import BaseModel
from fal.models.factory import ModelAdapterFactory
from fal.models.framework_adapters.pytorch_adapter import PyTorchModelAdapter
from fal.models.framework_adapters.sklearn_adapter import SklearnModelAdapter

__all__ = [
    "BaseModel",
    "ModelAdapterFactory",
    "PyTorchModelAdapter",
    "SklearnModelAdapter",
]
