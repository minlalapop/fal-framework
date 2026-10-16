import torch
import numpy as np

from experiments.benchmark.mnist_protocol import (
    BenchmarkMLP,
    derive_client_seed,
    parameter_payload_bytes,
)
from experiments.benchmark.flower_art_run import _prepare_data
from fal.dataset import ArrayDatasetLoader, DatasetManager, IIDSplitter


def test_e5_protocol_derives_distinct_client_and_phase_seeds():
    fit_seed = derive_client_seed(42, client_id=1, server_round=2, phase="fit")
    evaluate_seed = derive_client_seed(42, client_id=1, server_round=2, phase="evaluate")
    other_client_seed = derive_client_seed(42, client_id=2, server_round=2, phase="fit")

    assert len({fit_seed, evaluate_seed, other_client_seed}) == 3


def test_e5_payload_counts_download_and_upload_per_client_round():
    model = BenchmarkMLP()
    parameter_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())

    assert parameter_payload_bytes(model, clients=3, rounds=5) == parameter_bytes * 30
    assert all(value.dtype == torch.float32 for value in model.state_dict().values())


def test_direct_baseline_reproduces_fal_partitions():
    X = np.arange(120).reshape(60, 2)
    y = np.arange(60) % 10
    direct_clients, direct_test = _prepare_data(X, y, seed=42)
    manager = DatasetManager(
        loader=ArrayDatasetLoader(X, y),
        splitter=IIDSplitter(shuffle_data=True, random_state=42),
        num_clients=3,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    ).prepare()

    for direct, fal in zip(direct_clients, manager.get_clients_data()):
        np.testing.assert_array_equal(direct[0], fal[0])
        np.testing.assert_array_equal(direct[1], fal[1])
    np.testing.assert_array_equal(direct_test[0], manager.get_test_data()[0])
    np.testing.assert_array_equal(direct_test[1], manager.get_test_data()[1])
