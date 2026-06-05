from fal.dataset import BuiltinDatasetLoader, DatasetManager, IIDSplitter


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
