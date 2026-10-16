from fal.models import ModelAdapterFactory
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


def test_model_adapter_factory_wraps_sklearn_like_estimators():
    estimator = FakeSklearnEstimator()
    model = ModelAdapterFactory.create(estimator)

    model.fit([[1], [2]], [0, 1])

    assert model.framework == "sklearn"
    assert estimator.was_fit is True
    assert model.predict([[3], [4]]) == [0, 0]


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
