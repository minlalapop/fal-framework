import numpy as np
import pytest

from fal.models import BaseModel, ModelAdapterFactory
from fal.evaluation.losses.losses import ModelLoss


class FakeSklearnEstimator:
    def __init__(self):
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        return [0 for _ in range(len(X))]


class CustomBackendModel:
    def __init__(self):
        self.parameters = np.array([0.0])


class CustomBackendAdapter(BaseModel):
    def __init__(self, model, marker=None):
        self.model = model
        self.marker = marker

    @property
    def framework(self):
        return "custom_backend"

    def fit(self, X, y, **params):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def get_raw_model(self):
        return self.model


def test_model_adapter_factory_wraps_sklearn_like_estimators():
    estimator = FakeSklearnEstimator()
    model = ModelAdapterFactory.create(estimator)

    model.fit([[1], [2]], [0, 1])

    assert model.framework == "sklearn"
    assert estimator.was_fit is True
    assert model.predict([[3], [4]]) == [0, 0]


def test_registered_model_adapter_is_automatically_resolved(request):
    ModelAdapterFactory.register(
        "custom_backend",
        CustomBackendAdapter,
        detector=lambda model: isinstance(model, CustomBackendModel),
    )
    request.addfinalizer(lambda: ModelAdapterFactory._registry.pop("custom_backend", None))
    request.addfinalizer(lambda: ModelAdapterFactory._detectors.pop("custom_backend", None))

    adapter = ModelAdapterFactory.create(
        CustomBackendModel(),
        {"marker": "used"},
    )

    assert isinstance(adapter, CustomBackendAdapter)
    assert adapter.marker == "used"


def test_ambiguous_model_detection_fails_and_existing_explicit_path_resolves_it(request):
    for name in ("first_backend", "second_backend"):
        ModelAdapterFactory.register(
            name,
            CustomBackendAdapter,
            detector=lambda model: isinstance(model, CustomBackendModel),
        )
        request.addfinalizer(
            lambda name=name: ModelAdapterFactory._registry.pop(name, None)
        )
        request.addfinalizer(
            lambda name=name: ModelAdapterFactory._detectors.pop(name, None)
        )

    raw_model = CustomBackendModel()
    with pytest.raises(TypeError, match="Ambiguous model adapter detection"):
        ModelAdapterFactory.create(raw_model)

    explicit = CustomBackendAdapter(raw_model, marker="explicit")
    assert ModelAdapterFactory.create(explicit) is explicit


def test_model_adapter_registration_rejects_incompatible_class_and_detector():
    with pytest.raises(TypeError, match="BaseModel"):
        ModelAdapterFactory.register("invalid_backend", object, detector=lambda model: True)
    with pytest.raises(TypeError, match="detector"):
        ModelAdapterFactory.register("missing_detector", CustomBackendAdapter)


def test_custom_model_backend_uses_existing_flower_parameter_adapter_contract(request):
    from fal.fl.flower.flower_client.flower_parameters import (
        FlowerParameterAdapter,
        ParameterAdapterFactory,
    )

    class CustomParameterAdapter(FlowerParameterAdapter):
        def supports(self, model):
            return model.framework == "custom_backend"

        def get_parameters(self, model):
            return [model.get_raw_model().parameters.copy()]

        def set_parameters(self, model, parameters):
            model.get_raw_model().parameters = np.asarray(parameters[0]).copy()

    ParameterAdapterFactory.register("custom_backend", CustomParameterAdapter)
    request.addfinalizer(
        lambda: ParameterAdapterFactory._registry.pop("custom_backend", None)
    )
    model = CustomBackendAdapter(CustomBackendModel())
    adapter = ParameterAdapterFactory.create(model)

    adapter.set_parameters(model, [np.array([3.0])])

    assert adapter.get_parameters(model)[0].tolist() == [3.0]


def test_pytorch_adapter_predicts_in_batches():
    import torch
    from torch import nn

    raw_model = nn.Linear(3, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "batch_size": 2,
            "predict_batch_size": 2,
            "device": "cpu",
        },
    )

    predictions = model.predict(torch.zeros(5, 3))

    assert predictions.shape == (5,)
    assert raw_model.training is False


def test_pytorch_adapter_fit_restores_training_mode_after_evaluation():
    import torch
    from torch import nn

    raw_model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4, 2))
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "batch_size": 2,
            "device": "cpu",
        },
    )
    X = torch.zeros(4, 2)
    y = torch.tensor([0, 1, 0, 1])

    model.predict(X)
    assert raw_model.training is False

    model.fit(X, y, epochs=1)
    assert raw_model.training is True


def test_art_attack_uses_evaluation_mode_without_updating_batchnorm_state():
    import numpy as np
    import torch
    from torch import nn

    from fal.adv.art.art_backend import ARTBackend

    raw_model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Linear(2, 2))
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "batch_size": 4,
            "device": "cpu",
        },
    )
    X = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    running_mean = raw_model[0].running_mean.clone()
    raw_model.train()

    adversarial = ARTBackend().generate("fgsm", model, X, y, {"eps": 0.05})

    assert adversarial.shape == X.shape
    assert raw_model.training is False
    assert torch.equal(raw_model[0].running_mean, running_mean)


def test_pytorch_adapter_defaults_to_cpu():
    import torch
    from torch import nn

    raw_model = nn.Linear(2, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
        },
    )

    assert model.device.type == "cpu"
    assert next(raw_model.parameters()).device.type == "cpu"


def test_pytorch_adapter_rejects_unavailable_cuda():
    import pytest
    import torch
    from torch import nn

    if torch.cuda.is_available():
        pytest.skip("This validation requires a CPU-only test host.")

    raw_model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="CUDA is not available"):
        ModelAdapterFactory.create(
            raw_model,
            {
                "criterion": nn.CrossEntropyLoss(),
                "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
                "device": "cuda",
            },
        )


def test_art_uses_same_device_type_as_pytorch_model():
    import numpy as np
    import torch
    from torch import nn

    from fal.adv.art.pytorch_art_adapter import PyTorchARTAdapter

    raw_model = nn.Linear(2, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "device": "cpu",
        },
    )
    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    adapter = PyTorchARTAdapter()
    adapter.classifier_cls = FakeClassifier
    adapter.create_estimator(
        model,
        {},
        X=np.zeros((2, 2), dtype=np.float32),
        y=np.array([0, 1]),
    )

    assert captured["device_type"] == "cpu"
    assert next(raw_model.parameters()).device.type == "cpu"


def test_art_default_clip_values_remain_zero_one_for_bounded_data():
    import numpy as np
    import torch
    from torch import nn

    from fal.adv.art.pytorch_art_adapter import PyTorchARTAdapter

    raw_model = nn.Linear(2, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "device": "cpu",
        },
    )
    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    adapter = PyTorchARTAdapter()
    adapter.classifier_cls = FakeClassifier
    adapter.create_estimator(
        model,
        {},
        X=np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
        y=np.array([0, 1]),
    )

    assert captured["clip_values"] == (0.0, 1.0)


def test_art_does_not_infer_batch_dependent_bounds_outside_zero_one():
    import numpy as np
    import torch
    from torch import nn

    from fal.adv.art.pytorch_art_adapter import PyTorchARTAdapter

    raw_model = nn.Linear(2, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "device": "cpu",
        },
    )
    captured = []

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.append(kwargs["clip_values"])

    adapter = PyTorchARTAdapter()
    adapter.classifier_cls = FakeClassifier
    adapter.create_estimator(
        model,
        {},
        X=np.array([[-2.0, -1.0], [-1.5, -0.5]], dtype=np.float32),
        y=np.array([0, 1]),
    )
    adapter.create_estimator(
        model,
        {},
        X=np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        y=np.array([0, 1]),
    )

    assert captured == [None, None]


def test_explicit_per_feature_clip_values_are_stable_across_batches_and_models():
    import numpy as np

    from fal.adv.art.art_backend import ARTBackend
    from fal.adv.art.art_model_adapter import ARTModelAdapter, validate_clip_values

    lower = np.array([-5.0, 10.0])
    upper = np.array([0.0, 20.0])
    captured = []

    class RecordingAdapter(ARTModelAdapter):
        def create_estimator(self, model, config, X=None, y=None):
            captured.append(validate_clip_values(config["clip_values"], X))
            return object()

    backend = ARTBackend(
        art_model_adapter=RecordingAdapter(),
        estimator_config={"clip_values": (lower, upper)},
    )
    batches = (
        np.array([[-4.0, 11.0], [-3.0, 12.0]]),
        np.array([[-2.0, 18.0], [-1.0, 19.0]]),
    )

    backend.build_estimator(object(), X=batches[0], y=np.array([0, 1]))
    backend.build_estimator(object(), X=batches[1], y=np.array([1, 0]))

    assert len(captured) == 2
    for actual_lower, actual_upper in captured:
        assert np.array_equal(actual_lower, lower)
        assert np.array_equal(actual_upper, upper)


def test_adversarial_training_and_evaluation_share_explicit_art_bounds():
    from fal.adv.art.art_backend import ARTBackend
    from fal.adv.art.art_model_adapter import ARTModelAdapter, validate_clip_values
    from fal.adv.attacks.evasion.fgsm import FGSM
    from fal.evaluation.metrics import MetricFactory
    from fal.trainers import AdversarialTrainer

    bounds = (np.array([-5.0, 10.0]), np.array([0.0, 20.0]))
    captured = []

    class RecordingAdapter(ARTModelAdapter):
        def create_estimator(self, model, config, X=None, y=None):
            captured.append(validate_clip_values(config["clip_values"], X))
            return object()

    class PassthroughAttack:
        def __init__(self, estimator, **params):
            pass

        def generate(self, x, y=None):
            return np.asarray(x).copy()

    class Model:
        def fit(self, X, y, **params):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    backend = ARTBackend(
        art_model_adapter=RecordingAdapter(),
        estimator_config={"clip_values": bounds},
    )
    backend.FastGradientMethod = PassthroughAttack
    trainer = AdversarialTrainer(
        model=Model(),
        attack=FGSM(backend, {"eps": 0.1}),
        metrics_manager=MetricFactory.create(["accuracy"]),
    )
    data = (
        np.array([[-4.0, 11.0], [-1.0, 19.0]]),
        np.array([0, 1]),
    )

    trainer.train(data)
    trainer.evaluate(data)

    assert len(captured) == 2
    for actual_lower, actual_upper in captured:
        assert np.array_equal(actual_lower, bounds[0])
        assert np.array_equal(actual_upper, bounds[1])


def test_sklearn_art_adapter_validates_and_forwards_explicit_clip_values():
    import numpy as np

    from fal.adv.art.sklearn_art_adapter import SklearnARTAdapter

    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class WrappedModel:
        def get_raw_model(self):
            return object()

    adapter = SklearnARTAdapter()
    adapter.classifier_cls = FakeClassifier
    adapter.create_estimator(
        WrappedModel(),
        {"clip_values": (np.array([-2.0, 10.0]), np.array([2.0, 20.0]))},
        X=np.array([[-1.0, 12.0], [1.0, 18.0]]),
        y=np.array([0, 1]),
    )

    assert np.array_equal(captured["clip_values"][0], [-2.0, 10.0])
    assert np.array_equal(captured["clip_values"][1], [2.0, 20.0])


@pytest.mark.parametrize(
    ("clip_values", "X", "message"),
    [
        ((0.0,), np.zeros((2, 2)), "pair"),
        ((1.0, 0.0), np.zeros((2, 2)), "strictly below"),
        ((0.0, np.inf), np.zeros((2, 2)), "finite"),
        ((np.zeros(3), np.ones(3)), np.zeros((2, 2)), "broadcastable"),
        ((0.0, 1.0), np.array([[0.0, 2.0]]), "outside"),
    ],
)
def test_art_clip_values_reject_invalid_configuration(clip_values, X, message):
    import pytest

    from fal.adv.art.art_model_adapter import validate_clip_values

    with pytest.raises(ValueError, match=message):
        validate_clip_values(clip_values, X)


def test_art_backend_rejects_non_mapping_estimator_configuration():
    from fal.adv.art.art_backend import ARTBackend

    with pytest.raises(ValueError, match="estimator_config.*dictionary"):
        ARTBackend(estimator_config=[])


@pytest.mark.parametrize(
    ("attack_name", "attack_params"),
    [
        ("fgsm", {"eps": 0.05}),
        ("pgd", {"eps": 0.05, "eps_step": 0.02, "max_iter": 2, "batch_size": 4}),
    ],
)
def test_art_zero_one_default_matches_explicit_bounds_and_respects_budget(
    attack_name,
    attack_params,
):
    import numpy as np
    import torch
    from torch import nn

    from fal.adv.art.art_backend import ARTBackend

    torch.manual_seed(42)
    raw_model = nn.Linear(2, 2)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "device": "cpu",
        },
    )
    X = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])

    np.random.seed(42)
    historical = ARTBackend().generate(attack_name, model, X, y, attack_params)
    np.random.seed(42)
    explicit = ARTBackend(
        estimator_config={"clip_values": (0.0, 1.0)}
    ).generate(attack_name, model, X, y, attack_params)

    assert np.allclose(historical, explicit)
    assert np.max(np.abs(explicit - X)) <= attack_params["eps"] + 1e-6
    assert np.min(explicit) >= 0.0
    assert np.max(explicit) <= 1.0


def test_pytorch_model_loss_batches_forward_and_preserves_exact_criterion():
    import pytest
    import torch
    from torch import nn

    class RecordingLinear(nn.Linear):
        def __init__(self):
            super().__init__(3, 2, bias=False)
            self.batch_sizes = []

        def forward(self, values):
            self.batch_sizes.append(len(values))
            return super().forward(values)

    raw_model = RecordingLinear()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0]),
        ignore_index=-100,
    )
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": criterion,
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.01),
            "batch_size": 2,
            "predict_batch_size": 2,
            "device": "cpu",
        },
    )
    X = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    y = torch.tensor([0, 1, -100, 1, 0])

    with torch.no_grad():
        expected = float(criterion(raw_model(X), y).item())
    raw_model.batch_sizes.clear()

    actual = ModelLoss().compute(model, X, y)

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert raw_model.batch_sizes == [2, 2, 1]


def test_pytorch_fedprox_reuses_one_reference_across_multiple_fit_calls():
    import torch
    from torch import nn

    torch.manual_seed(42)
    raw_model = nn.Linear(1, 2, bias=False)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.1),
            "batch_size": 2,
            "device": "cpu",
        },
    )
    X = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([0, 1])
    initial = raw_model.weight.detach().clone()

    model._begin_local_training({"proximal_mu": 0.5})
    reference = model._local_training_reference
    model.fit(X, y, epochs=1, batch_size=2, proximal_mu=0.5)
    after_first_fit = raw_model.weight.detach().clone()
    model.fit(X, y, epochs=1, batch_size=2, proximal_mu=0.5)

    assert reference is model._local_training_reference
    assert torch.equal(reference[0], initial)
    assert not torch.equal(after_first_fit, initial)
    assert not torch.equal(raw_model.weight.detach(), after_first_fit)
    model._end_local_training()
    assert model._local_training_reference is None


def test_pytorch_proximal_term_and_gradient_match_independent_calculation():
    import pytest
    import torch
    from torch import nn

    raw_model = nn.Linear(1, 1, bias=False)
    raw_model.weight.data.fill_(2.0)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.MSELoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.1),
            "device": "cpu",
        },
    )
    reference = [torch.tensor([[0.5]])]
    mu = 0.4
    base_loss = (raw_model.weight - 3.0).pow(2).sum()
    actual_loss = model._add_proximal_term(base_loss, reference, mu)
    actual_loss.backward()

    independent_weight = torch.tensor([[2.0]], requires_grad=True)
    expected_loss = (
        (independent_weight - 3.0).pow(2).sum()
        + (mu / 2.0) * (independent_weight - reference[0]).pow(2).sum()
    )
    expected_loss.backward()

    assert actual_loss.detach().item() == pytest.approx(
        expected_loss.detach().item()
    )
    assert raw_model.weight.grad.item() == pytest.approx(
        independent_weight.grad.item()
    )


def test_pytorch_zero_mu_adds_no_proximal_penalty():
    import torch
    from torch import nn

    raw_model = nn.Linear(1, 1, bias=False)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.MSELoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.1),
            "device": "cpu",
        },
    )
    base_loss = raw_model.weight.pow(2).sum()

    actual = model._add_proximal_term(
        base_loss,
        [torch.full_like(raw_model.weight, 100.0)],
        proximal_mu=0.0,
    )

    assert actual is base_loss


def test_pytorch_fedprox_reference_is_renewed_between_rounds():
    import torch
    from torch import nn

    raw_model = nn.Linear(1, 1, bias=False)
    model = ModelAdapterFactory.create(
        raw_model,
        {
            "criterion": nn.MSELoss(),
            "optimizer": torch.optim.SGD(raw_model.parameters(), lr=0.1),
            "device": "cpu",
        },
    )

    model._begin_local_training({"proximal_mu": 0.5})
    first_reference = model._local_training_reference[0].clone()
    model._end_local_training()
    with torch.no_grad():
        raw_model.weight.add_(1.0)
    model._begin_local_training({"proximal_mu": 0.5})
    second_reference = model._local_training_reference[0].clone()

    assert not torch.equal(first_reference, second_reference)
    assert torch.equal(second_reference, raw_model.weight.detach())
