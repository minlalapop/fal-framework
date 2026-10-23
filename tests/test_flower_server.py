from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters
from flwr.common import ConfigRecord

from fal.fl.flower.flower_client.flower_client_adapter import FAL_TELEMETRY_PREFIX
from fal.fl.flower.flower_server.flower_server_adapter import (
    _consume_client_telemetry,
    _configure_ray_accelerator_environment,
    _finish_open_rounds,
    _owned_ray_cpu_budget,
    _raise_if_all_clients_failed,
    _ray_client_resources,
    _validate_client_devices,
)
from fal.fl.flower.output import flower_output
from fal.fl.flower.flower_strategy.fedadam_strategy import FedAdamStrategyAdapter
from fal.fl.flower.flower_strategy.fedavg_strategy import FedAvgStrategyAdapter
from fal.fl.flower.flower_strategy.fedprox_strategy import FedProxStrategyAdapter
from fal.fl.flower.flower_strategy.flower_strategy_adapter import (
    _weighted_average_metrics,
)


class FakeResponse:
    def __init__(self, metrics):
        self.metrics = metrics


def test_server_extracts_telemetry_from_flower_config_record():
    metrics = ConfigRecord({
        "accuracy": 0.8,
        f"{FAL_TELEMETRY_PREFIX}client_id": 2,
        f"{FAL_TELEMETRY_PREFIX}local_training_time": 0.25,
    })
    client_timings = {}

    _consume_client_telemetry(
        [(None, FakeResponse(metrics))],
        server_round=3,
        phase="fit",
        client_timings=client_timings,
    )

    assert client_timings[(3, 2)] == {
        "round": 3,
        "client_id": 2,
        "local_training_time": 0.25,
    }
    assert dict(metrics) == {"accuracy": 0.8}


def test_server_records_round_wall_clock_time_without_summing_clients():
    durations = {}

    _finish_open_rounds({1: 10.0}, durations, finished_at=12.5)

    assert durations == {1: 2.5}


def test_server_stops_all_client_failure_and_preserves_original_cause():
    cause = ValueError("original client failure")

    with pytest.raises(RuntimeError, match="round 4.*all 3") as raised:
        _raise_if_all_clients_failed(
            results=[],
            failures=[cause, RuntimeError("second"), RuntimeError("third")],
            server_round=4,
            phase="training",
        )

    assert raised.value.__cause__ is cause


def test_server_preserves_partial_failure_policy_and_empty_disabled_phase():
    _raise_if_all_clients_failed(
        results=[object()],
        failures=[ValueError("one client failed")],
        server_round=2,
        phase="training",
    )
    _raise_if_all_clients_failed(
        results=[],
        failures=[],
        server_round=2,
        phase="evaluation",
    )


def test_flower_output_does_not_hide_unrelated_python_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with flower_output(verbose=1):
            warnings.warn("visible warning", UserWarning)

    assert [str(item.message) for item in caught] == ["visible warning"]


def test_ray_resources_never_create_more_actors_than_clients():
    resources = _ray_client_resources(
        num_clients=3,
        cluster_resources={"CPU": 12.0},
    )

    assert resources == {"num_cpus": 4.0}


def test_ray_resources_allow_sequential_execution_from_external_limit():
    resources = _ray_client_resources(
        num_clients=3,
        cluster_resources={"CPU": 1.0},
    )

    assert resources == {"num_cpus": 1.0}


def test_ray_resources_limit_concurrency_and_reserve_fractional_gpu():
    resources = _ray_client_resources(
        num_clients=4,
        cluster_resources={"CPU": 8.0, "GPU": 1.0},
        num_cpus=1.0,
        num_gpus=0.5,
        max_concurrent_clients=2,
    )

    assert resources == {"num_cpus": 4.0, "num_gpus": 0.5}


def test_ray_resources_reject_unavailable_gpu():
    with pytest.raises(ValueError, match="Ray exposes only 0"):
        _ray_client_resources(
            num_clients=2,
            cluster_resources={"CPU": 2.0},
            num_gpus=1.0,
        )


def test_ray_resources_reject_invalid_direct_request():
    with pytest.raises(ValueError, match="num_cpus"):
        _ray_client_resources(
            num_clients=2,
            cluster_resources={"CPU": 2.0},
            num_cpus=0.0,
        )


def test_owned_ray_cpu_budget_limits_actor_concurrency(monkeypatch):
    monkeypatch.setattr(
        "fal.fl.flower.flower_server.flower_server_adapter.os.cpu_count",
        lambda: 12,
    )

    assert _owned_ray_cpu_budget(3, 1.0, None) == 3
    assert _owned_ray_cpu_budget(3, 1.0, 1) == 1


def test_cuda_device_requires_ray_gpu_reservation():
    client = SimpleNamespace(
        model=SimpleNamespace(
            framework="pytorch",
            device=SimpleNamespace(type="cuda", index=None),
        ),
        secondary_model=None,
    )

    with pytest.raises(ValueError, match="requires `num_gpus`"):
        _validate_client_devices([client], num_gpus=0.0)


def test_positive_ray_gpu_reservation_requires_cuda_model():
    client = SimpleNamespace(
        model=SimpleNamespace(
            framework="pytorch",
            device=SimpleNamespace(type="cpu", index=None),
        ),
        secondary_model=None,
    )

    with pytest.raises(ValueError, match="every FAL PyTorch model"):
        _validate_client_devices([client], num_gpus=0.5)


def test_mps_is_rejected_by_ray_simulation_with_explicit_message():
    client = SimpleNamespace(
        model=SimpleNamespace(
            framework="pytorch",
            device=SimpleNamespace(type="mps", index=None),
        ),
        secondary_model=None,
    )

    with pytest.raises(ValueError, match="MPS is not supported"):
        _validate_client_devices([client], num_gpus=0.0)


def test_ray_accelerator_warning_uses_targeted_environment_switch(monkeypatch):
    monkeypatch.delenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", raising=False)

    _configure_ray_accelerator_environment()

    assert __import__("os").environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"


def test_evaluation_losses_are_weighted_by_each_clients_evaluated_examples():
    aggregated = _weighted_average_metrics([
        (2, {"clean_accuracy": 0.5, "clean_loss": 2.0}),
        (6, {"clean_accuracy": 1.0, "clean_loss": 1.0}),
    ])

    assert aggregated["clean_accuracy"] == pytest.approx(0.875)
    assert aggregated["clean_loss"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("adapter", "expected_type"),
    [
        (FedAvgStrategyAdapter(), "FedAvg"),
        (FedProxStrategyAdapter({"proximal_mu": 0.1}), "FedProx"),
        (FedAdamStrategyAdapter(), "FedAdam"),
    ],
)
def test_all_fl_strategies_use_the_same_evaluation_metric_aggregation(
    adapter,
    expected_type,
):
    adapter.set_initial_parameters(ndarrays_to_parameters([np.array([0.0])]))
    strategy = adapter.build_strategy()

    assert type(strategy).__name__ == expected_type
    aggregated = strategy.evaluate_metrics_aggregation_fn([
        (1, {"attack_accuracy": 0.0, "attack_loss": 2.0}),
        (3, {"attack_accuracy": 1.0, "attack_loss": 1.0}),
    ])
    assert aggregated == {
        "attack_accuracy": pytest.approx(0.75),
        "attack_loss": pytest.approx(1.25),
    }
