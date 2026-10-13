"""Configuration object for FAL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SUPPORTED_ATTACKS = {"fgsm", "pgd", "deepfool"}
SUPPORTED_ATTACK_BACKENDS = {"art"}
SUPPORTED_TRAINERS = {"clean", "attack_only", "adversarial", "comparison", "mixed"}
SUPPORTED_METRICS = {
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "roc_auc",
    "confusion_matrix",
    "classification_report",
}
SUPPORTED_LOSSES = {"log_loss", "model_loss", "mse", "mean_squared_error"}
SUPPORTED_FL_BACKENDS = {"flower"}
SUPPORTED_FL_ALGORITHMS = {"fedavg", "fedprox", "fedadam"}


@dataclass
class FALConfig:
    """Central configuration for a Federated Adversarial Learning experiment.

    The public ``FAL`` facade should collect user arguments, store them here,
    and call ``validate`` before the builder creates concrete components.
    """

    dataset: Any | None = None
    X: Any | None = None
    y: Any | None = None
    target: str | None = None
    dataset_params: dict[str, Any] = field(default_factory=dict)

    model: Any | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    attack_type: str | None = None
    attack_params: dict[str, Any] = field(default_factory=dict)

    trainer_type: str = "clean"
    trainer_params: dict[str, Any] = field(default_factory=dict)

    metrics: Any = field(default_factory=lambda: ["accuracy"])
    loss: Any | None = None

    num_clients: int = 2
    num_rounds: int = 1
    test_size: float = 0.2
    iid: bool = True
    shards_per_client: int = 2
    random_state: int | None = None
    shuffle: bool = True

    fl_backend: str = "flower"
    fl_algorithm: str = "fedavg"
    fl_params: dict[str, Any] = field(default_factory=dict)

    attack_backend: str = "art"
    verbose: int = 1

    def __post_init__(self) -> None:
        self.attack_type = _normalize_optional_name(self.attack_type)
        self.attack_backend = _normalize_required_name(self.attack_backend, "attack_backend")
        self.trainer_type = _normalize_required_name(self.trainer_type, "trainer_type")
        self.fl_backend = _normalize_required_name(self.fl_backend, "fl_backend")
        self.fl_algorithm = _normalize_required_name(self.fl_algorithm, "fl_algorithm")
        self.metrics = _normalize_metrics_config(self.metrics)
        self.loss = _normalize_loss_config(self.loss)

    def validate(self) -> "FALConfig":
        """Validate and return this config for fluent builder usage."""

        self._validate_data_source()
        self._validate_model()
        self._validate_mappings()
        self._validate_training()
        self._validate_metrics()
        self._validate_loss()
        self._validate_federated_learning()
        self._validate_runtime()
        return self

    def _validate_data_source(self) -> None:
        uses_named_dataset = self.dataset is not None
        uses_array_dataset = self.X is not None or self.y is not None

        if not uses_named_dataset and not uses_array_dataset:
            raise ValueError("Provide either `dataset` or both `X` and `y`.")

        if uses_named_dataset and uses_array_dataset:
            raise ValueError("Provide either `dataset` or `X`/`y`, not both.")

        if uses_array_dataset:
            self._validate_array_dataset()

        if uses_named_dataset:
            self._validate_named_dataset()

    def _validate_array_dataset(self) -> None:
        if self.X is None or self.y is None:
            raise ValueError("Array input requires both `X` and `y`.")

    def _validate_named_dataset(self) -> None:
        if isinstance(self.dataset, str) and self.dataset.lower().endswith(".csv"):
            if not self.target:
                raise ValueError("CSV datasets require `target` to identify the label column.")

    def _validate_model(self) -> None:
        if self.model is None:
            raise ValueError("`model` is required.")

    def _validate_mappings(self) -> None:
        _ensure_mapping(self.dataset_params, "dataset_params")
        _ensure_mapping(self.model_params, "model_params")
        _ensure_mapping(self.attack_params, "attack_params")
        _ensure_mapping(self.trainer_params, "trainer_params")
        _ensure_mapping(self.fl_params, "fl_params")

    def _validate_training(self) -> None:
        if self.trainer_type not in SUPPORTED_TRAINERS:
            supported = ", ".join(sorted(SUPPORTED_TRAINERS))
            raise ValueError(f"Unsupported trainer_type `{self.trainer_type}`. Supported: {supported}.")

        if self.attack_type is not None and self.attack_type not in SUPPORTED_ATTACKS:
            supported = ", ".join(sorted(SUPPORTED_ATTACKS))
            raise ValueError(f"Unsupported attack_type `{self.attack_type}`. Supported: {supported}.")

        if self.attack_backend not in SUPPORTED_ATTACK_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_ATTACK_BACKENDS))
            raise ValueError(f"Unsupported attack_backend `{self.attack_backend}`. Supported: {supported}.")

        attack_trainers = {"attack_only", "adversarial", "comparison", "mixed"}
        if self.trainer_type in attack_trainers and self.attack_type is None:
            raise ValueError(f"`trainer_type='{self.trainer_type}'` requires `attack_type`.")

    def _validate_metrics(self) -> None:
        if not self.metrics:
            raise ValueError("At least one metric is required.")

        metric_names = [
            name
            for name in (_get_metric_name(metric) for metric in self.metrics)
            if name is not None
        ]
        unsupported = sorted(set(metric_names) - SUPPORTED_METRICS)
        if unsupported:
            supported = ", ".join(sorted(SUPPORTED_METRICS))
            names = ", ".join(unsupported)
            raise ValueError(f"Unsupported metrics: {names}. Supported: {supported}.")

    def _validate_loss(self) -> None:
        if self.loss is None:
            return

        if not isinstance(self.loss, (str, dict)):
            return

        loss_name = self.loss if isinstance(self.loss, str) else self.loss.get("name") or self.loss.get("loss")
        if loss_name not in SUPPORTED_LOSSES:
            supported = ", ".join(sorted(SUPPORTED_LOSSES))
            raise ValueError(f"Unsupported loss `{loss_name}`. Supported: {supported}.")

    def _validate_federated_learning(self) -> None:
        if not isinstance(self.num_clients, int) or self.num_clients <= 0:
            raise ValueError("`num_clients` must be a positive integer.")

        if not isinstance(self.num_rounds, int) or self.num_rounds <= 0:
            raise ValueError("`num_rounds` must be a positive integer.")

        if not 0 < self.test_size < 1:
            raise ValueError("`test_size` must be between 0 and 1.")

        if not isinstance(self.shards_per_client, int) or self.shards_per_client <= 0:
            raise ValueError("`shards_per_client` must be a positive integer.")

        if self.fl_backend not in SUPPORTED_FL_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_FL_BACKENDS))
            raise ValueError(f"Unsupported fl_backend `{self.fl_backend}`. Supported: {supported}.")

        if self.fl_algorithm not in SUPPORTED_FL_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_FL_ALGORITHMS))
            raise ValueError(f"Unsupported fl_algorithm `{self.fl_algorithm}`. Supported: {supported}.")

    def _validate_runtime(self) -> None:
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError("`verbose` must be a non-negative integer.")


def _normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_required_name(value, "name")


def _normalize_required_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return value.strip().lower().replace("-", "_")


def _ensure_mapping(value: Mapping[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"`{field_name}` must be a dictionary.")


def _normalize_loss_config(value: Any | None) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _normalize_required_name(value, "loss")
    if isinstance(value, dict):
        normalized = dict(value)
        key = "name" if "name" in normalized else "loss"
        if key in normalized:
            normalized[key] = _normalize_required_name(normalized[key], "loss")
        return normalized
    return value


def _normalize_metric_config(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_required_name(value, "metrics")
    if isinstance(value, dict):
        normalized = dict(value)
        key = "name" if "name" in normalized else "metric"
        if key in normalized:
            normalized[key] = _normalize_required_name(normalized[key], "metrics")
        return normalized
    return value


def _normalize_metrics_config(value: Any) -> list[Any]:
    if value is None:
        values = ["accuracy"]
    elif isinstance(value, (str, dict)):
        values = [value]
    else:
        values = list(value)

    return [_normalize_metric_config(metric) for metric in values]


def _get_metric_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("name") or value.get("metric")
    return None
