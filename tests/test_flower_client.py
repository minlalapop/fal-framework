import math

from fal.evaluation import EvaluationResult
from fal.fl.flower.flower_client.flower_client_adapter import (
    FAL_SERVER_ROUND_KEY,
    FAL_TELEMETRY_PREFIX,
    FlowerClientAdapter,
)
from fal.fl.flower.flower_backend import _available_losses


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
