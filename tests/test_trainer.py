import numpy as np

from fal.trainers import (
    AdversarialTrainer,
    AttackOnlyTrainer,
    CleanTrainer,
    MixedTrainer,
    TrainerFactory,
)


class FakeModel:
    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y, **params):
        self.fit_calls.append((np.asarray(X), np.asarray(y), dict(params)))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class FakeAttack:
    def generate(self, model, X, y):
        return np.asarray(X) + 10


class FakeMetricsManager:
    def compute_all(self, y_true, y_pred):
        return {
            "accuracy": 1.0,
            "f1": 1.0,
        }


class FakeLossManager:
    def compute(self, model, X, y):
        return 0.5


def test_clean_trainer_calls_model_fit():
    model = FakeModel()
    trainer = CleanTrainer(
        model=model,
        metrics_manager=FakeMetricsManager(),
        trainer_params={"epochs": 2},
    )

    trainer.train(([[1], [2]], [0, 1]))

    assert len(model.fit_calls) == 1
    assert model.fit_calls[0][2] == {"epochs": 2}


def test_attack_only_trainer_does_not_fit_by_default():
    model = FakeModel()
    trainer = AttackOnlyTrainer(
        model=model,
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        loss_manager=FakeLossManager(),
    )

    trainer.train(([[1], [2]], [0, 1]))

    assert model.fit_calls == []


def test_attack_only_trainer_can_clean_fit_before_attack_evaluation():
    model = FakeModel()
    trainer = AttackOnlyTrainer(
        model=model,
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        trainer_params={"clean_fit": True},
    )

    trainer.train(([[1], [2]], [0, 1]))

    assert len(model.fit_calls) == 1


def test_adversarial_trainer_trains_on_clean_and_adversarial_mix():
    model = FakeModel()
    trainer = AdversarialTrainer(
        model=model,
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        trainer_params={"epochs": 1},
    )

    trainer.train((np.array([[1], [2]]), np.array([0, 1])))

    X_fit, y_fit, params = model.fit_calls[0]
    assert X_fit.tolist() == [[1], [2], [11], [12]]
    assert y_fit.tolist() == [0, 1, 0, 1]
    assert params == {"epochs": 1}


def test_mixed_trainer_prefixes_clean_attack_and_trained_metrics():
    trainer = MixedTrainer(
        model=FakeModel(),
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        loss_manager=FakeLossManager(),
    )

    result = trainer.evaluate((np.array([[1], [2]]), np.array([0, 1])))

    assert result.loss == 0.5
    assert set(result.metrics) == {
        "clean_accuracy",
        "clean_f1",
        "attack_accuracy",
        "attack_f1",
        "trained_accuracy",
        "trained_f1",
    }


def test_trainer_factory_supports_comparison_alias_for_mixed_trainer():
    trainer = TrainerFactory.create(
        trainer_type="comparison",
        model=FakeModel(),
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
    )

    assert isinstance(trainer, MixedTrainer)
