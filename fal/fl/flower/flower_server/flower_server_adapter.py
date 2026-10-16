"""Flower server adapter."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
import logging
import math
import time
from types import MethodType
from typing import Any

import flwr as fl
import ray
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fal.fl.flower.metric_codec import decode_metric_value
from fal.fl.flower.flower_client.flower_client_adapter import (
    FAL_SERVER_ROUND_KEY,
    FAL_TELEMETRY_PREFIX,
)
from fal.fl.flower.output import flower_output
from fal.progress import FALProgressReporter


@dataclass
class FlowerServerResult:
    """Internal Flower backend result."""

    history: Any
    final_parameters: list[Any] | None = None
    timings: dict[str, Any] = field(default_factory=dict)


class FlowerServerAdapter:
    """Start a Flower simulation from prepared client adapters."""

    def start_server(
        self,
        clients: list[Any],
        strategy: Any,
        num_rounds: int = 1,
        verbose: int = 1,
        run_config: dict[str, Any] | None = None,
    ) -> Any:
        progress = FALProgressReporter(verbose=verbose)
        progress.start(
            num_rounds=num_rounds,
            num_clients=len(clients),
            run_config=run_config or {"fl_algorithm": _strategy_name(strategy)},
        )

        initial_parameters = ndarrays_to_parameters(clients[0].get_parameters({}))
        if hasattr(strategy, "set_initial_parameters"):
            strategy.set_initial_parameters(initial_parameters)

        flower_strategy = strategy.build_strategy() if hasattr(strategy, "build_strategy") else strategy
        final_parameters: list[Any] | None = None
        round_started_at: dict[int, float] = {}
        round_durations: dict[int, float] = {}
        client_timings: dict[tuple[int, int], dict[str, Any]] = {}

        if hasattr(flower_strategy, "configure_fit"):
            original_configure_fit = flower_strategy.configure_fit

            def configure_fit(
                strategy_self: Any,
                server_round: int,
                parameters: Any,
                client_manager: Any,
            ) -> Any:
                now = time.perf_counter()
                _finish_open_rounds(round_started_at, round_durations, now)
                round_started_at[server_round] = now
                progress.round_start(server_round)
                instructions = original_configure_fit(server_round, parameters, client_manager)
                _add_round_to_instructions(instructions, server_round)
                return instructions

            flower_strategy.configure_fit = MethodType(configure_fit, flower_strategy)

        original_aggregate_fit = flower_strategy.aggregate_fit

        def aggregate_fit(strategy_self: Any, server_round: int, results: list[Any], failures: list[Any]) -> Any:
            nonlocal final_parameters
            _consume_client_telemetry(
                results,
                server_round=server_round,
                phase="fit",
                client_timings=client_timings,
            )
            aggregated = original_aggregate_fit(server_round, results, failures)
            progress.fit_end(
                round_number=server_round,
                success_count=len(results),
                failure_count=len(failures),
            )
            parameters = aggregated[0] if aggregated else None
            if parameters is not None:
                final_parameters = parameters_to_ndarrays(parameters)
            return aggregated

        flower_strategy.aggregate_fit = MethodType(aggregate_fit, flower_strategy)

        if hasattr(flower_strategy, "configure_evaluate"):
            original_configure_evaluate = flower_strategy.configure_evaluate

            def configure_evaluate(
                strategy_self: Any,
                server_round: int,
                parameters: Any,
                client_manager: Any,
            ) -> Any:
                instructions = original_configure_evaluate(
                    server_round,
                    parameters,
                    client_manager,
                )
                _add_round_to_instructions(instructions, server_round)
                return instructions

            flower_strategy.configure_evaluate = MethodType(
                configure_evaluate,
                flower_strategy,
            )

        original_aggregate_evaluate = flower_strategy.aggregate_evaluate

        def aggregate_evaluate(
            strategy_self: Any,
            server_round: int,
            results: list[Any],
            failures: list[Any],
        ) -> Any:
            _consume_client_telemetry(
                results,
                server_round=server_round,
                phase="evaluate",
                client_timings=client_timings,
            )
            aggregated = original_aggregate_evaluate(server_round, results, failures)
            loss = aggregated[0] if aggregated else None
            metrics = aggregated[1] if aggregated else {}
            started_at = round_started_at.get(server_round)
            if started_at is not None:
                round_durations[server_round] = time.perf_counter() - started_at
            progress.evaluate_end(
                round_number=server_round,
                loss=_available_loss(loss),
                metrics={
                    name: decode_metric_value(value)
                    for name, value in metrics.items()
                },
            )
            return aggregated

        flower_strategy.aggregate_evaluate = MethodType(aggregate_evaluate, flower_strategy)

        owns_ray = not ray.is_initialized()
        try:
            if owns_ray:
                ray.init(
                    **_ray_init_args(verbose),
                    num_cpus=len(clients),
                )

            client_resources = _ray_client_resources(
                num_clients=len(clients),
                cluster_resources=ray.cluster_resources(),
            )
            client_refs = [ray.put(client) for client in clients]

            def client_fn(context: Any) -> Any:
                client_id = _client_id_from_context(context)
                client = ray.get(client_refs[client_id])
                return client.to_flower_client()

            with flower_output(verbose):
                history = fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=len(clients),
                    config=fl.server.ServerConfig(num_rounds=num_rounds),
                    strategy=flower_strategy,
                    ray_init_args=_ray_init_args(verbose),
                    keep_initialised=True,
                    client_resources=client_resources,
                )
        finally:
            if owns_ray and ray.is_initialized():
                ray.shutdown()
        _finish_open_rounds(
            round_started_at,
            round_durations,
            time.perf_counter(),
        )
        progress.finish()
        timings = {
            "rounds": [
                {
                    "round": round_number,
                    "round_execution_time": round_durations[round_number],
                }
                for round_number in sorted(round_durations)
            ],
            "clients": [
                client_timings[key]
                for key in sorted(client_timings)
            ],
        }
        return FlowerServerResult(
            history=history,
            final_parameters=final_parameters,
            timings=timings,
        )


def _add_round_to_instructions(instructions: Any, server_round: int) -> None:
    for _, operation in instructions or []:
        operation.config[FAL_SERVER_ROUND_KEY] = server_round


def _consume_client_telemetry(
    results: list[Any],
    server_round: int,
    phase: str,
    client_timings: dict[tuple[int, int], dict[str, Any]],
) -> None:
    for _, response in results:
        metrics = getattr(response, "metrics", None)
        if not isinstance(metrics, MutableMapping):
            continue

        telemetry = {
            key.removeprefix(FAL_TELEMETRY_PREFIX): metrics.pop(key)
            for key in list(metrics)
            if key.startswith(FAL_TELEMETRY_PREFIX)
        }
        if not telemetry:
            continue

        client_id = int(telemetry.pop("client_id", -1))
        telemetry.pop("server_round", None)
        record = client_timings.setdefault(
            (server_round, client_id),
            {"round": server_round, "client_id": client_id},
        )
        for name, value in telemetry.items():
            record[f"{phase}_{name}" if name == "seed" else name] = value


def _finish_open_rounds(
    round_started_at: dict[int, float],
    round_durations: dict[int, float],
    finished_at: float,
) -> None:
    for round_number, started_at in round_started_at.items():
        if round_number not in round_durations:
            round_durations[round_number] = finished_at - started_at


def _available_loss(loss: Any) -> float | None:
    if loss is None:
        return None
    value = float(loss)
    return None if math.isnan(value) else value


def _client_id_from_context(context: Any) -> int:
    if isinstance(context, str):
        return int(context)

    node_config = getattr(context, "node_config", None)
    if isinstance(node_config, dict):
        for key in ("partition-id", "partition_id", "client-id", "client_id"):
            if key in node_config:
                return int(node_config[key])

    return int(context)


def _ray_init_args(verbose: int) -> dict[str, Any]:
    args: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if verbose < 2:
        args["logging_level"] = logging.ERROR
        args["log_to_driver"] = False
    return args


def _ray_client_resources(
    num_clients: int,
    cluster_resources: dict[str, Any],
) -> dict[str, float]:
    available_cpus = max(1.0, float(cluster_resources.get("CPU", 1.0)))
    concurrent_clients = min(num_clients, max(1, int(available_cpus)))
    return {
        "num_cpus": available_cpus / concurrent_clients,
        "num_gpus": 0.0,
    }


def _strategy_name(strategy: Any) -> str:
    name = type(strategy).__name__
    return name.replace("StrategyAdapter", "").replace("Adapter", "")
