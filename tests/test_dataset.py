import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fal.builder import CoreBuilder
from fal.config import FALConfig
from fal.dataset import (
    ArrayDatasetLoader,
    BuiltinDatasetLoader,
    CustomCSVDatasetLoader,
    DatasetLoader,
    DatasetManager,
    IIDSplitter,
    NonIIDSplitter,
)


class RegisteredDatasetLoader(DatasetLoader):
    def __init__(self, offset=0):
        self.offset = offset

    def load(self):
        X = np.arange(60).reshape(30, 2) + self.offset
        y = np.tile(np.array([0, 1, 2]), 10)
        return X, y


def test_builtin_loader_registers_cifar10():
    assert "cifar10" in BuiltinDatasetLoader.SUPPORTED_DATASETS


def test_builtin_iris_loader_returns_features_and_labels():
    X, y = BuiltinDatasetLoader("iris").load()

    assert len(X) == len(y) == 150
    assert X.shape[1] == 4


def test_registered_dataset_loader_uses_existing_named_dataset_path(request):
    BuiltinDatasetLoader.register("registered_dataset", RegisteredDatasetLoader)
    request.addfinalizer(
        lambda: BuiltinDatasetLoader.SUPPORTED_DATASETS.pop("registered_dataset", None)
    )

    X, y = BuiltinDatasetLoader("registered_dataset", offset=5).load()

    assert X.shape == (30, 2)
    assert y.shape == (30,)
    assert X[0].tolist() == [5, 6]


def test_builder_accepts_an_existing_dataset_loader_instance():
    loader = RegisteredDatasetLoader(offset=3)
    config = FALConfig(dataset=loader, model=LogisticRegression())

    actual = CoreBuilder(config)._build_dataset_loader()

    assert actual is loader


def test_unknown_named_dataset_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported built-in dataset"):
        BuiltinDatasetLoader("unknown_dataset").load()


def test_existing_csv_loader_behavior_is_preserved(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("feature_a,feature_b,label\n1,2,0\n3,4,1\n", encoding="utf-8")

    X, y = CustomCSVDatasetLoader(path, target="label").load()

    assert X.columns.tolist() == ["feature_a", "feature_b"]
    assert X.to_numpy().tolist() == [[1, 2], [3, 4]]
    assert y.tolist() == [0, 1]


def test_dataset_registration_rejects_incompatible_loader():
    with pytest.raises(TypeError, match="DatasetLoader"):
        BuiltinDatasetLoader.register("invalid_dataset", object)


def test_dataset_manager_rejects_invalid_loader_output():
    class InvalidLoader(DatasetLoader):
        def load(self):
            return np.zeros((3, 2)), np.zeros(2)

    manager = DatasetManager(loader=InvalidLoader())

    with pytest.raises(ValueError, match="same number"):
        manager.prepare()


def test_dataset_manager_prepares_test_data_and_client_splits():
    manager = DatasetManager(
        loader=BuiltinDatasetLoader("iris"),
        splitter=IIDSplitter(shuffle_data=True, random_state=42),
        num_clients=3,
        test_size=0.2,
        random_state=42,
    ).prepare()

    clients_data = manager.get_clients_data()
    X_test, y_test = manager.get_test_data()

    assert len(clients_data) == 3
    assert len(X_test) == len(y_test) == 30
    assert sum(len(y_client) for _, y_client in clients_data) == 120


def test_dataset_manager_keeps_test_samples_out_of_client_training_partitions():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100) % 2

    first = DatasetManager(
        loader=ArrayDatasetLoader(X, y),
        splitter=IIDSplitter(shuffle_data=True, random_state=42),
        num_clients=3,
        test_size=0.2,
        random_state=42,
    ).prepare()
    second = DatasetManager(
        loader=ArrayDatasetLoader(X, y),
        splitter=IIDSplitter(shuffle_data=True, random_state=42),
        num_clients=3,
        test_size=0.2,
        random_state=42,
    ).prepare()

    train_ids = np.concatenate([data[0].reshape(-1) for data in first.get_clients_data()])
    test_ids = first.get_test_data()[0].reshape(-1)

    assert set(train_ids).isdisjoint(test_ids)
    assert sorted(np.concatenate([train_ids, test_ids]).tolist()) == list(range(100))
    for first_client, second_client in zip(first.get_clients_data(), second.get_clients_data()):
        assert np.array_equal(first_client[0], second_client[0])
        assert np.array_equal(first_client[1], second_client[1])
    assert np.array_equal(first.get_test_data()[0], second.get_test_data()[0])


def test_non_iid_splitter_creates_label_skewed_client_partitions():
    samples_per_class = 100
    num_classes = 10
    X = np.arange(samples_per_class * num_classes)
    y = np.repeat(np.arange(num_classes), samples_per_class)

    clients = NonIIDSplitter(
        shards_per_client=2,
        shuffle_data=True,
        random_state=42,
    ).split(X, y, num_clients=10)

    assert all(len(np.unique(y_client)) <= 2 for _, y_client in clients)


def test_non_iid_splitter_preserves_every_sample_exactly_once():
    X = np.arange(103)
    y = np.repeat(np.arange(10), [11, 10, 10, 10, 10, 10, 10, 10, 10, 12])

    clients = NonIIDSplitter(
        shards_per_client=2,
        shuffle_data=True,
        random_state=42,
    ).split(X, y, num_clients=10)

    assigned_ids = np.concatenate([X_client for X_client, _ in clients])

    assert all(len(X_client) > 0 for X_client, _ in clients)
    assert len(assigned_ids) == len(X)
    assert len(np.unique(assigned_ids)) == len(X)
    assert sorted(assigned_ids.tolist()) == X.tolist()
