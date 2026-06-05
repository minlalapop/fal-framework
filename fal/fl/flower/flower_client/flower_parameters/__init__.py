"""Flower parameter adapters."""

from fal.fl.flower.flower_client.flower_parameters.flower_parameters_adapter import (
    FlowerParameterAdapter,
)
from fal.fl.flower.flower_client.flower_parameters.flower_parameter_factory import (
    ParameterAdapterFactory,
)
from fal.fl.flower.flower_client.flower_parameters.flower_sklearn_parameters_adapter import (
    SklearnParameterAdapter,
)

__all__ = [
    "FlowerParameterAdapter",
    "ParameterAdapterFactory",
    "SklearnParameterAdapter",
]

try:
    from fal.fl.flower.flower_client.flower_parameters.flower_pytorch_parameters_adapter import (
        PyTorchParameterAdapter,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.append("PyTorchParameterAdapter")
