import numpy as np
import pytest

from fal.fl.flower.flower_client.flower_parameters.flower_sklearn_parameters_adapter import (
    SklearnParameterAdapter,
)


class WrappedSklearnModel:
    framework = "sklearn"

    def __init__(self, raw_model):
        self.raw_model = raw_model

    def get_raw_model(self):
        return self.raw_model


class CustomParameterModel:
    def __init__(self):
        self.parameters = [np.array([1.0, 2.0]), np.array([0.5])]
        self.initialized = False

    def get_fal_parameters(self):
        return self.parameters

    def set_fal_parameters(self, parameters):
        self.parameters = parameters

    def initialize_fal_parameters(self, X, y, classes=None):
        self.initialized = True


class UnsupportedSklearnModel:
    pass


def test_sklearn_parameter_adapter_supports_custom_parameter_hooks():
    raw_model = CustomParameterModel()
    model = WrappedSklearnModel(raw_model)
    adapter = SklearnParameterAdapter()

    adapter.initialize(model, X=[[1], [2]], y=[0, 1], classes=[0, 1])
    parameters = adapter.get_parameters(model)
    adapter.set_parameters(model, [np.array([3.0, 4.0]), np.array([1.5])])

    assert raw_model.initialized is True
    assert [parameter.tolist() for parameter in parameters] == [[1.0, 2.0], [0.5]]
    assert [parameter.tolist() for parameter in raw_model.parameters] == [[3.0, 4.0], [1.5]]


def test_sklearn_parameter_adapter_explains_unsupported_models():
    model = WrappedSklearnModel(UnsupportedSklearnModel())
    adapter = SklearnParameterAdapter()

    with pytest.raises(ValueError, match="get_fal_parameters"):
        adapter.get_parameters(model)
