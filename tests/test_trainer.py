import numpy as np

from fal.evaluation.metrics import MetricFactory
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


class SplitPredictionModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        values = np.asarray(X).reshape(len(X), -1)[:, 0]
        return np.where(values < 10, values.astype(int) % 2, 1)


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


def test_clean_trainer_returns_only_clean_metrics():
    trainer = CleanTrainer(
        model=FakeModel(),
        metrics_manager=FakeMetricsManager(),
        loss_manager=FakeLossManager(),
    )

    result = trainer.evaluate((np.array([[1], [2]]), np.array([0, 1])))

    assert result.loss == 0.5
    assert result.metrics == {
        "clean_accuracy": 1.0,
        "clean_f1": 1.0,
    }


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


def test_attack_only_trainer_returns_only_attack_metrics():
    trainer = AttackOnlyTrainer(
        model=FakeModel(),
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        loss_manager=FakeLossManager(),
    )

    result = trainer.evaluate((np.array([[1], [2]]), np.array([0, 1])))

    assert result.loss == 0.5
    assert result.metrics == {
        "attack_accuracy": 1.0,
        "attack_f1": 1.0,
    }


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


def test_adversarial_trainer_records_attack_generation_times(monkeypatch):
    ticks = iter([10.0, 10.25, 20.0, 20.5])
    monkeypatch.setattr(
        "fal.trainers.adversarial.time.perf_counter",
        lambda: next(ticks),
    )
    trainer = AdversarialTrainer(
        model=FakeModel(),
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
        loss_manager=FakeLossManager(),
    )

    trainer.train((np.array([[1], [2]]), np.array([0, 1])))
    trainer.evaluate((np.array([[1], [2]]), np.array([0, 1])))

    assert trainer.last_attack_generation_train_time == 0.25
    assert trainer.last_attack_generation_evaluation_time == 0.5


def test_adversarial_evaluation_returns_only_trained_clean_and_adv_metrics():
    model = SplitPredictionModel()
    metrics_manager = MetricFactory.create(
        ["accuracy", "precision", "recall", "f1"]
    )
    trainer = AdversarialTrainer(
        model=model,
        attack=FakeAttack(),
        metrics_manager=metrics_manager,
        loss_manager=FakeLossManager(),
    )
    X_test = np.array([[0], [1], [2], [3]])
    y_test = np.array([0, 1, 0, 1])

    result = trainer.evaluate((X_test, y_test))

    clean_predictions = np.array([0, 1, 0, 1])
    attack_predictions = np.ones(4, dtype=int)
    expected_clean = metrics_manager.compute_all(y_test, clean_predictions)
    expected_attack = metrics_manager.compute_all(y_test, attack_predictions)

    assert model.predict_calls == 1
    assert result.loss == 0.5
    for name in ("accuracy", "precision", "recall", "f1"):
        assert result.metrics[f"trained_clean_{name}"] == expected_clean[name]
        assert result.metrics[f"trained_adv_{name}"] == expected_attack[name]
    assert set(result.metrics) == {
        "trained_clean_accuracy",
        "trained_clean_precision",
        "trained_clean_recall",
        "trained_clean_f1",
        "trained_adv_accuracy",
        "trained_adv_precision",
        "trained_adv_recall",
        "trained_adv_f1",
    }


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
        "trained_clean_accuracy",
        "trained_clean_f1",
        "trained_adv_accuracy",
        "trained_adv_f1",
    }


def test_mixed_trainer_trains_clean_and_adversarial_models_independently():
    clean_model = FakeModel()
    adversarial_model = FakeModel()
    trainer = MixedTrainer(
        model=clean_model,
        adversarial_model=adversarial_model,
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
    )

    trainer.train((np.array([[1], [2]]), np.array([0, 1])))

    assert len(clean_model.fit_calls) == 1
    assert len(adversarial_model.fit_calls) == 1
    assert clean_model.fit_calls[0][0].tolist() == [[1], [2]]
    assert adversarial_model.fit_calls[0][0].tolist() == [[1], [2], [11], [12]]


def test_trainer_factory_supports_comparison_alias_for_mixed_trainer():
    trainer = TrainerFactory.create(
        trainer_type="comparison",
        model=FakeModel(),
        attack=FakeAttack(),
        metrics_manager=FakeMetricsManager(),
    )

    assert isinstance(trainer, MixedTrainer)
