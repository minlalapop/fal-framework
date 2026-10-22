"""Terminal output control for Flower simulations."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import logging
from typing import Iterator


@contextmanager
def flower_output(verbose: int) -> Iterator[None]:
    """Reduce external Flower/Ray logging unless verbose debugging is requested."""

    if verbose >= 2:
        with nullcontext():
            yield
        return

    logger_names = ("flwr", "ray")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous_levels = [logger.level for logger in loggers]
    previous_handler_levels = [
        [handler.level for handler in logger.handlers]
        for logger in loggers
    ]

    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
            for handler in logger.handlers:
                handler.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level, handler_levels in zip(
            loggers,
            previous_levels,
            previous_handler_levels,
        ):
            logger.setLevel(level)
            for handler, handler_level in zip(logger.handlers, handler_levels):
                handler.setLevel(handler_level)
