"""Terminal progress display for FAL runs."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, TextIO

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    tqdm = None


class FALProgressReporter:
    """Framework-style progress reporter for terminal runs."""

    def __init__(self, verbose: int = 1, stream: TextIO | None = None) -> None:
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.enabled = verbose >= 1
        self.colors = _Color(enabled=_supports_color(self.stream))
        self.progress_bar = None
        self.last_postfix = ""
        self.total_rounds = 0
        self.total_clients = 0
        self.started_at = 0.0
        self.round_started_at: dict[int, float] = {}
        self.fit_status: dict[int, str] = {}
        self.metric_names: list[str] = []
        self.show_loss = False
        self.header_printed = False

    def start(
        self,
        num_rounds: int,
        num_clients: int,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        self.total_rounds = num_rounds
        self.total_clients = num_clients
        self.started_at = time.perf_counter()
        run_config = run_config or {}

        self._write(self.colors.bold("FAL Training"))
        self._write(self.colors.dim("Federated Adversarial Learning"))
        self._write("")
        for line in _format_setup(run_config, num_rounds, num_clients, self.colors):
            self._write(line)
        self._write("")

        if self._use_progress_bar():
            self.progress_bar = tqdm(
                total=num_rounds,
                desc=self.colors.blue("Rounds"),
                unit="round",
                leave=True,
                file=self.stream,
                dynamic_ncols=True,
            )

    def round_start(self, round_number: int) -> None:
        if self.enabled:
            self.round_started_at[round_number] = time.perf_counter()

    def fit_end(self, round_number: int, success_count: int, failure_count: int) -> None:
        if not self.enabled:
            return

        self.round_started_at.setdefault(round_number, time.perf_counter())
        self.fit_status[round_number] = f"{success_count}/{success_count + failure_count}"

    def evaluate_end(
        self,
        round_number: int,
        loss: float | None,
        metrics: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        scalar_metrics = {
            name: value
            for name, value in metrics.items()
            if _is_scalar(value)
        }

        elapsed = time.perf_counter() - self.round_started_at.get(
            round_number,
            time.perf_counter(),
        )

        if not self.header_printed:
            self.metric_names = _ordered_metric_names(scalar_metrics)
            self.show_loss = loss is not None
            self._write(_format_header(self.metric_names, include_loss=self.show_loss))
            self.header_printed = True

        row = _format_round(
            round_number=round_number,
            total_rounds=self.total_rounds,
            clients=self.fit_status.get(round_number, "-"),
            loss=loss if self.show_loss else None,
            include_loss=self.show_loss,
            metric_names=self.metric_names,
            metrics=scalar_metrics,
            elapsed=elapsed,
        )
        self._write(row)

        if self.progress_bar is not None:
            self.last_postfix = _format_postfix(
                clients=self.fit_status.get(round_number, "-"),
                loss=loss,
            )
            self.progress_bar.set_postfix_str(
                self.last_postfix,
                refresh=False,
            )
            self.progress_bar.update(1)

    def finish(self) -> None:
        if not self.enabled:
            return

        if self.progress_bar is not None:
            self.progress_bar.close()

        elapsed = time.perf_counter() - self.started_at
        self._write(self.colors.green(f"done in {elapsed:.1f}s"))
        self._write("")

    def _write(self, message: str) -> None:
        if self.progress_bar is not None:
            self.progress_bar.write(message)
            return
        print(message, file=self.stream, flush=True)

    def _use_progress_bar(self) -> bool:
        return (
            tqdm is not None
            and callable(getattr(self.stream, "isatty", None))
            and self.stream.isatty()
        )


class _Color:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def bold(self, value: str) -> str:
        return self._wrap(value, "1")

    def dim(self, value: str) -> str:
        return self._wrap(value, "2")

    def blue(self, value: str) -> str:
        return self._wrap(value, "34")

    def green(self, value: str) -> str:
        return self._wrap(value, "32")

    def cyan(self, value: str) -> str:
        return self._wrap(value, "36")

    def _wrap(self, value: str, code: str) -> str:
        if not self.enabled:
            return value
        return f"\033[{code}m{value}\033[0m"


def _format_setup(
    run_config: dict[str, Any],
    num_rounds: int,
    num_clients: int,
    colors: _Color,
) -> list[str]:
    strategy = run_config.get("fl_algorithm") or run_config.get("strategy") or "-"
    trainer = run_config.get("trainer_type") or "-"
    metrics = run_config.get("metrics") or []
    loss = run_config.get("loss")
    attack = run_config.get("attack")
    model = run_config.get("model")
    dataset = run_config.get("dataset")

    lines = [
        _setup_line(
            colors.cyan("FL"),
            f"strategy={_display_name(strategy)} | clients={num_clients} | rounds={num_rounds}",
        ),
        _setup_line(colors.cyan("Trainer"), _display_name(trainer)),
    ]

    if attack:
        lines.append(_setup_line(colors.cyan("Attack"), _format_attack(attack)))
    if model:
        lines.append(_setup_line(colors.cyan("Model"), str(model)))
    if dataset:
        lines.append(_setup_line(colors.cyan("Dataset"), str(dataset)))
    if metrics:
        lines.append(_setup_line(colors.cyan("Metrics"), ", ".join(map(str, metrics))))
    if loss:
        lines.append(_setup_line(colors.cyan("Loss"), str(loss)))

    return lines


def _setup_line(label: str, value: str) -> str:
    return f"{label:<12} {value}"


def _format_attack(attack: dict[str, Any]) -> str:
    name = _display_name(attack.get("type", "-"))
    params = attack.get("params") or {}
    if not params:
        return name
    rendered_params = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"{name}({rendered_params})"


def _display_name(value: Any) -> str:
    text = str(value)
    known_names = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedadam": "FedAdam",
        "fgsm": "FGSM",
        "pgd": "PGD",
        "deepfool": "DeepFool",
    }
    return known_names.get(text.lower(), text.replace("_", " ").title())


def _format_header(metric_names: list[str], include_loss: bool = True) -> str:
    columns = ["round", "clients"]
    if include_loss:
        columns.append("loss")
    columns.extend([*metric_names, "time"])
    return _format_row(columns, _column_widths(metric_names, include_loss))


def _format_round(
    round_number: int,
    total_rounds: int,
    clients: str,
    loss: float | None,
    include_loss: bool,
    metric_names: list[str],
    metrics: dict[str, Any],
    elapsed: float,
) -> str:
    values = [
        f"{round_number}/{total_rounds}",
        clients,
    ]
    if include_loss:
        values.append("-" if loss is None else f"{loss:.4f}")
    values.extend(_format_metric(metrics.get(name)) for name in metric_names)
    values.append(f"{elapsed:.1f}s")
    return _format_row(values, _column_widths(metric_names, include_loss))


def _format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(widths[index] if index < len(widths) else 8)
        for index, value in enumerate(values)
    ).rstrip()


def _format_postfix(
    clients: str,
    loss: float | None,
) -> str:
    values = [f"clients={clients}"]
    if loss is not None:
        values.append(f"loss={loss:.4f}")
    return " ".join(values)


def _column_widths(metric_names: list[str], include_loss: bool = True) -> list[int]:
    widths = [7, 7]
    if include_loss:
        widths.append(8)
    widths.extend([max(8, len(name)) for name in metric_names])
    widths.append(7)
    return widths


def _format_metric(value: Any) -> str:
    if _is_scalar(value):
        return f"{float(value):.4f}"
    return "-"


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _supports_color(stream: TextIO) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return callable(getattr(stream, "isatty", None)) and stream.isatty()
