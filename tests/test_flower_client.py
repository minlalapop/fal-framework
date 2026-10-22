import math

import numpy as np
import pytest

from fal.evaluation import EvaluationResult
from fal.fl.flower.flower_client.flower_client_adapter import (
    FAL_SERVER_ROUND_KEY,
    FAL_TELEMETRY_PREFIX,
    FlowerClientAdapter,
)
from fal.fl.flower.flower_backend import _available_losses
from fal.trainers import MixedTrainer


class FakeParameterAdapter:
    def get_parameters(self, model):
        return [1.0]

    def set_parameters(self, model, params):
        model.parameters = list(params)


class FakeTrainer:
    last_attack_generation_train_time = 0.25
    last_attack_generation_evaluation_time = 0.5

    def __init__(self, loss=None):
        self.loss = loss
        self.trainer_params = {}

    def train(self, data):
        return None

    def evaluate(self, data):
        return EvaluationResult(loss=self.loss, metrics={"accuracy": 0.75})


def _client(loss=None, client_id=2, random_state=42):
    return FlowerClientAdapter(
        model=type("Model", (), {})(),
        trainer=FakeTrainer(loss=loss),
        parameter_adapter=FakeParameterAdapter(),
        data=([1, 2], [0, 1]),
        test_data=([3, 4], [0, 1]),
        client_id=client_id,
        random_state=random_state,
    )


def test_missing_loss_uses_nan_marker_for_flower():
    loss, _, _ = _client(loss=None).evaluate(
        [1.0],
        {FAL_SERVER_ROUND_KEY: 1},
    )

    assert math.isnan(loss)


def test_real_loss_is_preserved_for_flower():
    loss, _, _ = _client(loss=0.625).evaluate(
        [1.0],
        {FAL_SERVER_ROUND_KEY: 1},
    )

    assert loss == 0.625


def test_prefixed_evaluation_losses_are_transported_as_metrics_without_generic_loss():
    client = _client(loss=None)
    client.trainer.evaluate = lambda data: EvaluationResult(
        metrics={"clean_accuracy": 0.75, "clean_loss": 0.625},
    )

    loss, num_examples, metrics = client.evaluate(
        [1.0],
        {FAL_SERVER_ROUND_KEY: 1},
    )

    assert math.isnan(loss)
    assert num_examples == 2
    assert metrics["clean_accuracy"] == 0.75
    assert metrics["clean_loss"] == 0.625


def test_client_transports_two_independent_parameter_sets_for_mixed():
    client = _client()
    secondary_model = type("Model", (), {})()
    secondary_adapter = FakeParameterAdapter()
    client.secondary_model = secondary_model
    client.secondary_parameter_adapter = secondary_adapter

    assert client.get_parameters() == [1.0, 1.0]

    client.set_parameters([2.0, 3.0])

    assert client.model.parameters == [2.0]
    assert secondary_model.parameters == [3.0]


def test_flower_backend_removes_only_missing_loss_markers():
    losses = _available_losses([(1, math.nan), (2, 0.625)])

    assert losses == [(2, 0.625)]


def test_client_reports_local_timings_separately_from_model_metrics():
    client = _client(loss=0.5)

    _, _, fit_metrics = client.fit([1.0], {FAL_SERVER_ROUND_KEY: 3})
    _, _, evaluate_metrics = client.evaluate([1.0], {FAL_SERVER_ROUND_KEY: 3})

    assert fit_metrics[f"{FAL_TELEMETRY_PREFIX}local_training_time"] >= 0.0
    assert fit_metrics[f"{FAL_TELEMETRY_PREFIX}attack_generation_train_time"] == 0.25
    assert evaluate_metrics[f"{FAL_TELEMETRY_PREFIX}evaluation_time"] >= 0.0
    assert evaluate_metrics[f"{FAL_TELEMETRY_PREFIX}attack_generation_evaluation_time"] == 0.5
    assert evaluate_metrics["accuracy"] == 0.75


def test_client_keeps_training_and_evaluation_datasets_separate():
    trainer = FakeTrainer(loss=0.5)
    calls = {}

    def train(data):
        calls["train"] = data

    def evaluate(data):
        calls["evaluate"] = data
        return EvaluationResult(loss=0.5, metrics={"accuracy": 0.75})

    trainer.train = train
    trainer.evaluate = evaluate
    client = FlowerClientAdapter(
        model=type("Model", (), {})(),
        trainer=trainer,
        parameter_adapter=FakeParameterAdapter(),
        data=([1, 2], [0, 1]),
        test_data=([3, 4], [1, 0]),
    )

    client.fit([1.0], {FAL_SERVER_ROUND_KEY: 1})
    client.evaluate([1.0], {FAL_SERVER_ROUND_KEY: 1})

    assert calls["train"] == ([1, 2], [0, 1])
    assert calls["evaluate"] == ([3, 4], [1, 0])


def test_client_does_not_turn_evaluation_exceptions_into_metrics():
    client = _client()

    def fail(data):
        raise RuntimeError("attack generation failed")

    client.trainer.evaluate = fail

    with pytest.raises(RuntimeError, match="attack generation failed"):
        client.evaluate([1.0], {FAL_SERVER_ROUND_KEY: 1})


def test_client_seeds_are_stable_and_distinct(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "fal.fl.flower.flower_client.flower_client_adapter.seed_everything",
        lambda seed, deterministic=False: recorded.append((seed, deterministic)),
    )
    first = _client(client_id=0)
    second = _client(client_id=1)

    first.fit([1.0], {FAL_SERVER_ROUND_KEY: 2})
    second.fit([1.0], {FAL_SERVER_ROUND_KEY: 2})
    first.fit([1.0], {FAL_SERVER_ROUND_KEY: 2})

    assert recorded[0][0] != recorded[1][0]
    assert recorded[0][0] == recorded[2][0]


def test_client_configures_model_device_inside_flower_worker():
    model = type("Model", (), {"configure_runtime_device": lambda self: setattr(self, "ready", True)})()
    client = FlowerClientAdapter(
        model=model,
        trainer=FakeTrainer(),
        parameter_adapter=FakeParameterAdapter(),
        data=([1], [0]),
    )

    client.to_flower_client()

    assert model.ready is True


def test_fit_config_reaches_mixed_training_subtrainers_and_is_restored():
    class RecordingModel:
        def __init__(self):
            self.fit_calls = []

        def fit(self, X, y, **params):
            self.fit_calls.append(dict(params))
            return self

    class IdentityAttack:
        def generate(self, model, X, y):
            return np.asarray(X)

    initial_params = {}
    clean_model = RecordingModel()
    adversarial_model = RecordingModel()
    trainer = MixedTrainer(
        model=clean_model,
        adversarial_model=adversarial_model,
        attack=IdentityAttack(),
        trainer_params=initial_params,
    )
    client = FlowerClientAdapter(
        model=clean_model,
        trainer=trainer,
        parameter_adapter=FakeParameterAdapter(),
        data=(np.array([[1.0], [2.0]]), np.array([0, 1])),
    )

    client._train_with_config({"runtime_option": 7})

    assert clean_model.fit_calls == [{"runtime_option": 7}]
    assert adversarial_model.fit_calls == [{"runtime_option": 7}]
    assert trainer.trainer_params is initial_params
    assert trainer.clean_trainer.trainer_params is initial_params
    assert trainer.adversarial_trainer.trainer_params is initial_params
    assert initial_params == {}
    assert trainer.attack_trainer.trainer_params == {"clean_fit": False}


def test_fit_config_restores_standalone_trainer_params_after_success_and_error():
    class RecordingTrainer:
        def __init__(self, should_raise=False):
            self.trainer_params = {"epochs": 2}
            self.should_raise = should_raise
            self.received = None

        def train(self, data):
            self.received = dict(self.trainer_params)
            if self.should_raise:
                raise RuntimeError("training failed")

    for should_raise in (False, True):
        trainer = RecordingTrainer(should_raise=should_raise)
        original_params = trainer.trainer_params
        client = FlowerClientAdapter(
            model=type("Model", (), {})(),
            trainer=trainer,
            parameter_adapter=FakeParameterAdapter(),
            data=([1, 2], [0, 1]),
        )

        if should_raise:
            with pytest.raises(RuntimeError, match="training failed"):
                client._train_with_config({"epochs": 9, "runtime_option": 7})
        else:
            client._train_with_config({"epochs": 9, "runtime_option": 7})

        assert trainer.received == {"epochs": 9, "runtime_option": 7}
        assert trainer.trainer_params is original_params
        assert trainer.trainer_params == {"epochs": 2}


def test_client_manages_independent_local_training_references_for_mixed():
    class LifecycleModel:
        def __init__(self):
            self.parameters = []
            self.reference = None
            self.fit_references = []

        def _begin_local_training(self, config):
            self.reference = tuple(self.parameters)

        def _end_local_training(self):
            self.reference = None

        def fit(self, X, y, **params):
            self.fit_references.append((self.reference, dict(params)))
            return self

    class IdentityAttack:
        def generate(self, model, X, y):
            return np.asarray(X)

    clean_model = LifecycleModel()
    adversarial_model = LifecycleModel()
    trainer = MixedTrainer(
        model=clean_model,
        adversarial_model=adversarial_model,
        attack=IdentityAttack(),
    )
    client = FlowerClientAdapter(
        model=clean_model,
        trainer=trainer,
        parameter_adapter=FakeParameterAdapter(),
        data=(np.array([[1.0], [2.0]]), np.array([0, 1])),
    )
    client.secondary_model = adversarial_model
    client.secondary_parameter_adapter = FakeParameterAdapter()

    client.fit([11.0, 22.0], {"proximal_mu": 0.5})

    assert clean_model.fit_references == [((11.0,), {"proximal_mu": 0.5})]
    assert adversarial_model.fit_references == [((22.0,), {"proximal_mu": 0.5})]
    assert clean_model.reference is None
    assert adversarial_model.reference is None


def test_client_cleans_local_training_reference_after_exception():
    class LifecycleModel:
        def __init__(self):
            self.reference = None

        def _begin_local_training(self, config):
            self.reference = "active"

        def _end_local_training(self):
            self.reference = None

    class RaisingTrainer(FakeTrainer):
        def train(self, data):
            raise ValueError("client failed")

    model = LifecycleModel()
    client = FlowerClientAdapter(
        model=model,
        trainer=RaisingTrainer(),
        parameter_adapter=FakeParameterAdapter(),
        data=([1], [0]),
    )

    with pytest.raises(ValueError, match="client failed"):
        client.fit([1.0], {"proximal_mu": 0.5})

    assert model.reference is None
