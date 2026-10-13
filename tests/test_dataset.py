import numpy as np

from fal.dataset import BuiltinDatasetLoader, DatasetManager, IIDSplitter, NonIIDSplitter


def test_builtin_loader_registers_cifar10():
    assert "cifar10" in BuiltinDatasetLoader.SUPPORTED_DATASETS


def test_builtin_iris_loader_returns_features_and_labels():
    X, y = BuiltinDatasetLoader("iris").load()

    assert len(X) == len(y) == 150
    assert X.shape[1] == 4


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
