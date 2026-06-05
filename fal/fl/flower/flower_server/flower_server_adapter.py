"""Flower server adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import MethodType
from typing import Any

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fal.fl.flower.metric_codec import decode_metric_value
from fal.fl.flower.output import flower_output
from fal.progress import FALProgressReporter


@dataclass
class FlowerServerResult:
    """Internal Flower backend result."""

    history: Any
    final_parameters: list[Any] | None = None


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

        if hasattr(flower_strategy, "configure_fit"):
            original_configure_fit = flower_strategy.configure_fit

            def configure_fit(
                strategy_self: Any,
                server_round: int,
                parameters: Any,
                client_manager: Any,
            ) -> Any:
                progress.round_start(server_round)
                return original_configure_fit(server_round, parameters, client_manager)

            flower_strategy.configure_fit = MethodType(configure_fit, flower_strategy)

        original_aggregate_fit = flower_strategy.aggregate_fit

        def aggregate_fit(strategy_self: Any, server_round: int, results: list[Any], failures: list[Any]) -> Any:
            nonlocal final_parameters
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

        original_aggregate_evaluate = flower_strategy.aggregate_evaluate

        def aggregate_evaluate(
            strategy_self: Any,
            server_round: int,
            results: list[Any],
            failures: list[Any],
        ) -> Any:
            aggregated = original_aggregate_evaluate(server_round, results, failures)
            loss = aggregated[0] if aggregated else None
            metrics = aggregated[1] if aggregated else {}
            progress.evaluate_end(
                round_number=server_round,
                loss=loss,
                metrics={
                    name: decode_metric_value(value)
                    for name, value in metrics.items()
                },
            )
            return aggregated

        flower_strategy.aggregate_evaluate = MethodType(aggregate_evaluate, flower_strategy)

        def client_fn(context: Any) -> Any:
            client_id = _client_id_from_context(context)
            return clients[client_id].to_flower_client()

        with flower_output(verbose):
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(clients),
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=flower_strategy,
                ray_init_args=_ray_init_args(verbose),
            )
        progress.finish()
        return FlowerServerResult(history=history, final_parameters=final_parameters)


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


def _strategy_name(strategy: Any) -> str:
    name = type(strategy).__name__
    return name.replace("StrategyAdapter", "").replace("Adapter", "")
