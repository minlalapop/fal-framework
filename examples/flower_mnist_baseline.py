
from __future__ import annotations

import copy
import os
import random
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import datasets, transforms


TRAINER_TYPE = os.getenv("FLOWER_BASELINE_TRAINER", "adversarial").strip().lower()
NUM_CLIENTS = 3
NUM_ROUNDS = 5
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001
EPS = 0.025
SEED = 42
DEVICE = "cpu"


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class RecordingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters = aggregated[0] if aggregated else None
        if parameters is not None:
            self.final_parameters = fl.common.parameters_to_ndarrays(parameters)
        return aggregated


class MNISTClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        initial_state: OrderedDict,
        train_data: tuple[np.ndarray, np.ndarray],
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.cid = cid
        self.model = SimpleMLP().to(DEVICE)
        self.model.load_state_dict(copy.deepcopy(initial_state))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config=None):
        return [value.detach().cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict(
            {
                key: torch.tensor(value)
                for key, value in zip(keys, parameters)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        X_train, y_train = self.train_data

        if TRAINER_TYPE == "clean":
            self._fit_clean(X_train, y_train)
        elif TRAINER_TYPE == "attack_only":
            self._fit_clean(X_train, y_train)
        elif TRAINER_TYPE == "adversarial":
            self._fit_adversarial(X_train, y_train)
        else:
            raise ValueError(f"Unsupported FLOWER_BASELINE_TRAINER: {TRAINER_TYPE}")

        return self.get_parameters(config), len(y_train), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        X_test, y_test = self.test_data

        if TRAINER_TYPE == "clean":
            loss, metrics = self._evaluate_clean(X_test, y_test)
        elif TRAINER_TYPE == "attack_only":
            loss, metrics = self._evaluate_attack_only(X_test, y_test)
        elif TRAINER_TYPE == "adversarial":
            loss, metrics = self._evaluate_adversarial_mix(X_test, y_test)
        else:
            raise ValueError(f"Unsupported FLOWER_BASELINE_TRAINER: {TRAINER_TYPE}")

        return loss, len(y_test), metrics

    def _fit_clean(self, X: np.ndarray, y: np.ndarray) -> None:
        dataset = torch.utils.data.TensorDataset(
            _to_tensor(X),
            _to_target_tensor(y),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.model.train()
        for _ in range(EPOCHS):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def _fit_adversarial(self, X: np.ndarray, y: np.ndarray) -> None:
        X_adv = self._generate_adversarial(X, y)
        X_mix = np.concatenate([X, X_adv], axis=0)
        y_mix = np.concatenate([y, y], axis=0)
        self._fit_clean(X_mix, y_mix)

    def _evaluate_clean(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
        loss, y_pred = self._loss_and_predictions(X, y)
        return loss, _compute_metrics(y, y_pred)

    def _evaluate_attack_only(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
        X_adv = self._generate_adversarial(X, y)
        loss, y_pred = self._loss_and_predictions(X_adv, y)
        return loss, _compute_metrics(y, y_pred)

    def _evaluate_adversarial_mix(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
        X_adv = self._generate_adversarial(X, y)
        X_mix = np.concatenate([X, X_adv], axis=0)
        y_mix = np.concatenate([y, y], axis=0)
        loss, y_pred = self._loss_and_predictions(X_mix, y_mix)
        return loss, _compute_metrics(y_mix, y_pred)

    def _generate_adversarial(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classifier = PyTorchClassifier(
            model=self.model,
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
            clip_values=(0.0, 1.0),
        )
        attack = FastGradientMethod(estimator=classifier, eps=EPS)
        return attack.generate(x=_writable_array(X), y=_writable_array(y))

    def _loss_and_predictions(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        X_tensor = _to_tensor(X).to(DEVICE)
        y_tensor = _to_target_tensor(y).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            y_pred = outputs.argmax(dim=-1).detach().cpu().numpy()

        return float(loss.detach().cpu().item()), y_pred


def main() -> None:
    _set_seed(SEED)
    X, y = _load_mnist_arrays()
    clients_data, test_data = _prepare_fal_like_data(X, y)

    base_model = SimpleMLP().to(DEVICE)
    initial_state = copy.deepcopy(base_model.state_dict())
    clients = [
        MNISTClient(
            cid=cid,
            initial_state=initial_state,
            train_data=clients_data[cid],
            test_data=test_data,
        )
        for cid in range(NUM_CLIENTS)
    ]

    strategy = RecordingFedAvg(
        evaluate_metrics_aggregation_fn=_weighted_average_metrics,
    )

    def client_fn(context: Any):
        client_id = _client_id_from_context(context)
        return clients[client_id].to_client()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print(history)
    print("baseline trainer:", TRAINER_TYPE)
    print("baseline losses:", history.losses_distributed)
    print("baseline metrics:", history.metrics_distributed)


def _load_mnist_arrays() -> tuple[np.ndarray, np.ndarray]:
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    X = torch.stack([sample for sample, _ in dataset]).numpy()
    y = np.asarray([label for _, label in dataset])
    return X, y


def _prepare_fal_like_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    indices = list(range(len(y_train)))
    random.Random(SEED).shuffle(indices)
    client_indices = [indices[index::NUM_CLIENTS] for index in range(NUM_CLIENTS)]
    clients_data = [
        (X_train[client_index], y_train[client_index])
        for client_index in client_indices
    ]
    return clients_data, (X_test, y_test)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _weighted_average_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_examples: dict[str, int] = {}

    for num_examples, client_metrics in metrics:
        for name, value in client_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            totals[name] = totals.get(name, 0.0) + float(value) * num_examples
            total_examples[name] = total_examples.get(name, 0) + num_examples

    return {
        name: total / total_examples[name]
        for name, total in totals.items()
        if total_examples[name] > 0
    }


def _to_tensor(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(_writable_array(values))
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor


def _to_target_tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(_writable_array(values)).long()


def _writable_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return array


def _client_id_from_context(context: Any) -> int:
    if isinstance(context, str):
        return int(context)

    node_config = getattr(context, "node_config", None)
    if isinstance(node_config, dict):
        for key in ("partition-id", "partition_id", "client-id", "client_id"):
            if key in node_config:
                return int(node_config[key])

    return int(context)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
