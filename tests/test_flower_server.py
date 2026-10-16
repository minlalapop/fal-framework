from flwr.common.record.configrecord import ConfigRecord

from fal.fl.flower.flower_client.flower_client_adapter import FAL_TELEMETRY_PREFIX
from fal.fl.flower.flower_server.flower_server_adapter import (
    _consume_client_telemetry,
    _finish_open_rounds,
    _ray_client_resources,
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


def test_ray_resources_never_create_more_actors_than_clients():
    resources = _ray_client_resources(
        num_clients=3,
        cluster_resources={"CPU": 12.0},
    )

    assert resources == {"num_cpus": 4.0, "num_gpus": 0.0}


def test_ray_resources_allow_sequential_execution_from_external_limit():
    resources = _ray_client_resources(
        num_clients=3,
        cluster_resources={"CPU": 1.0},
    )

    assert resources == {"num_cpus": 1.0, "num_gpus": 0.0}
