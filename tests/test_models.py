from fal.models import ModelAdapterFactory


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
