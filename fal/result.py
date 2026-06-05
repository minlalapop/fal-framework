"""Public result returned by FAL runs."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any


@dataclass
class FALResult:
    """Framework-level result independent from the concrete FL backend."""

    model: Any
    history: Any = None
    losses: list[tuple[int, float]] = field(default_factory=list)
    metrics: dict[str, list[tuple[int, Any]]] = field(default_factory=dict)
    backend: str | None = None
    raw_result: Any = None
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @property
    def raw_model(self) -> Any:
        if hasattr(self.model, "get_raw_model"):
            return self.model.get_raw_model()
        return self.model

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self.raw_model, file)

    def save_outputs(self, directory: str = "results", save_model: bool = True) -> dict[str, str]:
        output_dir = _next_run_directory(Path(directory))
        _clear_known_outputs(output_dir)

        paths: dict[str, str] = {"directory": str(output_dir)}

        if save_model:
            model_path = output_dir / "model.pkl"
            self.save(str(model_path))
            paths["model"] = str(model_path)

        metrics_path = output_dir / "metrics.csv"
        _write_metrics_csv(metrics_path, self.losses, self.metrics)
        paths["metrics"] = str(metrics_path)

        _, detailed_metrics = _split_metrics(self.metrics)
        if detailed_metrics:
            detailed_path = output_dir / "detailed_metrics.csv"
            _write_detailed_metrics_csv(detailed_path, detailed_metrics)
            paths["detailed_metrics"] = str(detailed_path)

        self.artifact_paths = paths
        return paths

    @property
    def latest_loss(self) -> float | None:
        if not self.losses:
            return None
        return self.losses[-1][1]

    @property
    def latest_metrics(self) -> dict[str, Any]:
        return {
            name: values[-1][1]
            for name, values in self.metrics.items()
            if values
        }

    def metric(self, name: str, round_number: int | None = None) -> Any:
        values = self.metrics.get(name)
        if not values:
            raise KeyError(f"Metric `{name}` is not available.")

        if round_number is None:
            return values[-1][1]

        for current_round, value in values:
            if current_round == round_number:
                return value

        raise KeyError(f"Metric `{name}` is not available for round {round_number}.")

    def summary(self, include_details: bool = True) -> str:
        lines = ["FAL Evaluation", "=============="]

        rounds = _round_numbers(self.losses, self.metrics)
        if rounds:
            lines.append(f"Rounds: {len(rounds)}")

        scalar_metrics, detailed_metrics = _split_metrics(self.metrics)
        table = _format_run_table(self.losses, scalar_metrics)
        if table:
            lines.extend(["", "Evaluation", table])

        if include_details and detailed_metrics:
            details = self.details()
            if details:
                lines.extend(["", details])

        return "\n".join(lines)

    def print_summary(self, include_details: bool = True) -> None:
        print(self.summary(include_details=include_details))

    def details(self) -> str:
        _, detailed_metrics = _split_metrics(self.metrics)
        if not detailed_metrics:
            return ""

        lines = ["Detailed Metrics", "----------------"]
        for name, values in detailed_metrics.items():
            round_number, value = values[-1]
            lines.extend(["", _title(name, round_number), _format_detail(name, value)])
        return "\n".join(lines)

    def print_details(self) -> None:
        details = self.details()
        if details:
            print(details)

    def __str__(self) -> str:
        return self.summary(include_details=False)


def _split_metrics(
    metrics: dict[str, list[tuple[int, Any]]],
) -> tuple[dict[str, list[tuple[int, Any]]], dict[str, list[tuple[int, Any]]]]:
    scalar_metrics = {}
    artifact_metrics = {}

    for name, values in metrics.items():
        if all(_is_scalar(value) for _, value in values):
            scalar_metrics[name] = values
        else:
            artifact_metrics[name] = values

    return scalar_metrics, artifact_metrics


def _clear_known_outputs(output_dir: Path) -> None:
    for filename in ("model.pkl", "metrics.csv", "detailed_metrics.csv"):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def _next_run_directory(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    run_numbers = [
        int(path.name.removeprefix("results_"))
        for path in base_dir.iterdir()
        if path.name.startswith("results_")
        and path.name.removeprefix("results_").isdigit()
    ]
    next_number = max(run_numbers, default=0) + 1
    run_dir = base_dir / f"results_{next_number}"
    run_dir.mkdir()
    return run_dir


def _write_metrics_csv(
    path: Path,
    losses: list[tuple[int, float]],
    metrics: dict[str, list[tuple[int, Any]]],
) -> None:
    scalar_metrics, _ = _split_metrics(metrics)
    metric_names = _ordered_metric_names(scalar_metrics)
    rounds = _round_numbers(losses, metrics)
    losses_by_round = {round_number: loss for round_number, loss in losses}
    values_by_metric = {
        name: {round_number: value for round_number, value in values}
        for name, values in scalar_metrics.items()
    }

    header = ["round"]
    if losses:
        header.append("loss")
    header.extend(metric_names)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for round_number in rounds:
            row: dict[str, Any] = {"round": round_number}
            if losses:
                row["loss"] = losses_by_round.get(round_number, "")
            for name in metric_names:
                row[name] = values_by_metric[name].get(round_number, "")
            writer.writerow(row)


def _write_detailed_metrics_csv(
    path: Path,
    metrics: dict[str, list[tuple[int, Any]]],
) -> None:
    metric_names = _ordered_metric_names(metrics)
    rounds = _round_numbers([], metrics)
    values_by_metric = {
        name: {round_number: value for round_number, value in values}
        for name, values in metrics.items()
    }
    header = ["round", *metric_names]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for round_number in rounds:
            row: dict[str, Any] = {"round": round_number}
            for name in metric_names:
                value = values_by_metric[name].get(round_number)
                row[name] = "" if value is None else _json_dumps(value)
            writer.writerow(row)


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=False)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def _round_numbers(
    losses: list[tuple[int, float]],
    metrics: dict[str, list[tuple[int, Any]]],
) -> list[int]:
    return sorted({
        round_number
        for round_number, _ in losses
    } | {
        round_number
        for values in metrics.values()
        for round_number, _ in values
    })


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_run_table(
    losses: list[tuple[int, float]],
    metrics: dict[str, list[tuple[int, Any]]],
) -> str:
    if not losses and not metrics:
        return ""

    metric_names = _ordered_metric_names(metrics)
    rounds = _round_numbers(losses, metrics)
    losses_by_round = {round_number: loss for round_number, loss in losses}
    values_by_metric = {
        name: {round_number: value for round_number, value in values}
        for name, values in metrics.items()
    }

    rows = []
    header = ["round"]
    if losses:
        header.append("loss")
    header.extend(metric_names)
    rows.append(header)

    for round_number in rounds:
        row = [str(round_number)]
        if losses:
            loss = losses_by_round.get(round_number)
            row.append("" if loss is None else _format_number(loss))
        for name in metric_names:
            value = values_by_metric[name].get(round_number)
            row.append("" if value is None else _format_number(value))
        rows.append(row)

    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(header))
    ]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _title(name: str, round_number: int) -> str:
    readable = _readable_metric_name(name)
    return f"{readable} (round {round_number})"


def _format_detail(name: str, value: Any) -> str:
    base_name = _base_metric_name(name)
    if base_name == "confusion_matrix":
        return _format_matrix(value)
    if base_name == "classification_report":
        return _format_classification_report(value)
    return _format_value(value)


def _format_matrix(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return _format_value(value)

    rows = [[str(item) for item in row] for row in value]
    labels = [str(index) for index in range(len(rows))]
    header = ["", *labels]
    table = [header, *[[labels[index], *row] for index, row in enumerate(rows)]]
    widths = [
        max(len(row[column]) for row in table)
        for column in range(len(header))
    ]
    return "\n".join(
        "  ".join(item.rjust(widths[index]) for index, item in enumerate(row))
        for row in table
    )


def _format_classification_report(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)

    rows = [["label", "precision", "recall", "f1", "support"]]
    for label, stats in value.items():
        if isinstance(stats, dict):
            rows.append([
                str(label),
                _format_number(stats.get("precision")),
                _format_number(stats.get("recall")),
                _format_number(stats.get("f1-score")),
                _format_number(stats.get("support"), decimals=0),
            ])
        elif label == "accuracy":
            rows.append(["accuracy", "", "", _format_number(stats), ""])

    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(rows[0]))
    ]
    return "\n".join(
        "  ".join(item.ljust(widths[index]) for index, item in enumerate(row))
        for row in rows
    )


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(row) for row in value)
    if isinstance(value, dict):
        return pformat(value, sort_dicts=False)
    return str(value)


def _format_number(value: Any, decimals: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    return ""


def _ordered_metric_names(metrics: dict[str, Any]) -> list[str]:
    prefixes = ("clean_", "attack_", "trained_")

    def key(name: str) -> tuple[int, str]:
        for index, prefix in enumerate(prefixes):
            if name.startswith(prefix):
                return index, name.removeprefix(prefix)
        return len(prefixes), name

    return sorted(metrics, key=key)


def _base_metric_name(name: str) -> str:
    for prefix in ("clean_", "attack_", "trained_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _readable_metric_name(name: str) -> str:
    phase_names = {
        "clean_": "Clean ",
        "attack_": "Attack ",
        "trained_": "Trained ",
    }
    for prefix, label in phase_names.items():
        if name.startswith(prefix):
            return label + name.removeprefix(prefix).replace("_", " ").title()
    return name.replace("_", " ").title()
