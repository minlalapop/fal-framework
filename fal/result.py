"""Public result returned by FAL runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import metadata
import json
import math
import platform
import pickle
from pathlib import Path
from pprint import pformat
import subprocess
import sys
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
    timings: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    config: Any = None

    def __post_init__(self) -> None:
        self.losses = [
            (round_number, loss)
            for round_number, loss in self.losses
            if not _is_unavailable_loss(loss)
        ]

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

        if self.timings:
            timings_path = output_dir / "timings.json"
            self.save_timings(str(timings_path))
            paths["timings"] = str(timings_path)

        if self.config is not None:
            config_path = output_dir / "config.json"
            self.save_config(str(config_path))
            paths["config"] = str(config_path)

        self.artifact_paths = paths
        return paths

    def save_timings(self, path: str) -> None:
        """Write timing telemetry to a JSON file."""
        timings_path = Path(path)
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        with timings_path.open("w", encoding="utf-8") as file:
            json.dump(_to_jsonable(self.timings), file, indent=2, ensure_ascii=False)

    def save_config(self, path: str) -> None:
        """Write a safe, reproducible description of the experiment."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(
                _configuration_payload(self.config),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

    def refresh_timings_output(self) -> None:
        """Refresh the timing artifact after the full public run completes."""
        timings_path = self.artifact_paths.get("timings")
        if timings_path:
            self.save_timings(timings_path)

    @property
    def latest_loss(self) -> float | None:
        if not self.losses:
            return None
        return self.losses[-1][1]

    @property
    def loss_available(self) -> bool:
        """Whether at least one real evaluation loss was recorded."""
        return bool(self.losses)

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
    for filename in (
        "model.pkl",
        "metrics.csv",
        "detailed_metrics.csv",
        "timings.json",
        "config.json",
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def _configuration_payload(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        values = {
            config_field.name: getattr(config, config_field.name)
            for config_field in fields(config)
        }
    elif isinstance(config, dict):
        values = dict(config)
    else:
        values = {"configuration": config}

    X = values.pop("X", None)
    y = values.pop("y", None)
    if X is not None or y is not None:
        values["dataset"] = "array"
        values["array_data"] = {
            "X": _array_metadata(X),
            "y": _array_metadata(y),
        }

    payload = {
        str(name): (
            "<redacted>"
            if _is_sensitive_key(name)
            else _config_jsonable(value)
        )
        for name, value in values.items()
    }
    payload["timing_unit"] = "seconds"
    payload["environment"] = _environment_metadata()
    payload["source"] = _source_metadata()
    return payload


def _config_jsonable(value: Any, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)

    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return {"type": _qualified_type_name(value), "recursive": True}

    if isinstance(value, dict):
        seen.add(identity)
        try:
            return {
                str(key): (
                    "<redacted>"
                    if _is_sensitive_key(key)
                    else _config_jsonable(item, seen)
                )
                for key, item in value.items()
            }
        finally:
            seen.remove(identity)

    if isinstance(value, (list, tuple, set)):
        seen.add(identity)
        try:
            return [_config_jsonable(item, seen) for item in value]
        finally:
            seen.remove(identity)

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _parameter_array(value)

    if hasattr(value, "defaults") and hasattr(value, "param_groups"):
        groups = []
        for group in value.param_groups:
            groups.append({
                str(name): _config_jsonable(item)
                for name, item in group.items()
                if name != "params" and not _is_sensitive_key(name)
            })
        return {
            "type": _qualified_type_name(value),
            "defaults": _config_jsonable(value.defaults),
            "param_groups": groups,
        }

    get_params = getattr(value, "get_params", None)
    if callable(get_params):
        try:
            params = get_params(deep=False)
        except (TypeError, ValueError):
            params = None
        if isinstance(params, dict):
            return {
                "type": _qualified_type_name(value),
                "params": _config_jsonable(params),
            }

    parameters = getattr(value, "parameters", None)
    state_dict = getattr(value, "state_dict", None)
    if callable(parameters) and callable(state_dict):
        try:
            parameter_count = sum(parameter.numel() for parameter in parameters())
        except (AttributeError, TypeError):
            parameter_count = None
        result = {
            "type": _qualified_type_name(value),
            "modules": [
                {
                    "name": name,
                    "type": _qualified_type_name(module),
                    "configuration": (
                        module.extra_repr()
                        if callable(getattr(module, "extra_repr", None))
                        else ""
                    ),
                }
                for name, module in value.named_modules()
            ] if callable(getattr(value, "named_modules", None)) else [],
        }
        if parameter_count is not None:
            result["parameter_count"] = int(parameter_count)
        return result

    if isinstance(value, type) or callable(value):
        return {"type": _qualified_type_name(value)}

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _config_jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass

    return {"type": _qualified_type_name(value)}


def _array_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        try:
            shape = (len(value),)
        except TypeError:
            shape = None
    return {
        "type": _qualified_type_name(value),
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
    }


def _parameter_array(value: Any) -> dict[str, Any]:
    result = _array_metadata(value) or {}
    size = getattr(value, "size", None)
    size = int(size) if isinstance(size, int) else None
    numel = getattr(value, "numel", None)
    if callable(numel):
        size = int(numel())
    if size is not None and size <= 256:
        serializable = value
        detach = getattr(serializable, "detach", None)
        if callable(detach):
            serializable = detach()
        cpu = getattr(serializable, "cpu", None)
        if callable(cpu):
            serializable = cpu()
        tolist = getattr(serializable, "tolist", None)
        if callable(tolist):
            result["values"] = tolist()
    return result


def _qualified_type_name(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    sensitive_names = {
        "access_key",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
    }
    return normalized in sensitive_names or any(
        normalized.endswith(f"_{name}")
        for name in sensitive_names
    )


def _environment_metadata() -> dict[str, Any]:
    packages = {
        "fal-framework": "fal-framework",
        "flower": "flwr",
        "ray": "ray",
        "pytorch": "torch",
        "art": "adversarial-robustness-toolbox",
        "numpy": "numpy",
        "scikit-learn": "scikit-learn",
    }
    versions = {}
    for name, distribution in packages.items():
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = None

    return {
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": versions,
    }


def _source_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit or None, "git_dirty": dirty}


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


def _is_unavailable_loss(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


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
    prefixes = ("clean_", "attack_", "trained_clean_", "trained_adv_", "trained_")
    metric_order = {
        "accuracy": 0,
        "loss": 1,
        "precision": 2,
        "recall": 3,
        "f1": 4,
    }

    def key(name: str) -> tuple[int, int, str]:
        for index, prefix in enumerate(prefixes):
            if name.startswith(prefix):
                base_name = name.removeprefix(prefix)
                return index, metric_order.get(base_name, len(metric_order)), base_name
        return len(prefixes), metric_order.get(name, len(metric_order)), name

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
