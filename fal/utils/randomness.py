"""Best-effort random seed control for reproducible FAL experiments."""

from __future__ import annotations

import random
from importlib import import_module
from typing import Any


_NUMPY_SEED_MODULUS = 2**32
_ROUND_SEED_OFFSET = 1_000_003
_CLIENT_SEED_OFFSET = 97_409
_EVALUATION_SEED_OFFSET = 53_123


def seed_everything(seed: int | None, deterministic: bool = False) -> None:
    """Seed available RNGs and optionally request deterministic PyTorch operations.

    Deterministic execution remains best effort because support depends on the
    selected hardware, PyTorch operations, and third-party libraries.
    """
    if seed is not None:
        random.seed(seed)

    np = _optional_import("numpy")
    if seed is not None and np is not None:
        np.random.seed(seed % _NUMPY_SEED_MODULUS)

    torch = _optional_import("torch")
    if torch is None:
        return

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_client_seed(
    base_seed: int | None,
    client_id: int,
    server_round: int,
    phase: str,
) -> int | None:
    """Derive a stable, distinct seed for a client, round, and phase."""
    if base_seed is None:
        return None

    phase_offset = _EVALUATION_SEED_OFFSET if phase == "evaluate" else 0
    return (
        base_seed
        + server_round * _ROUND_SEED_OFFSET
        + client_id * _CLIENT_SEED_OFFSET
        + phase_offset
    ) % _NUMPY_SEED_MODULUS


def _optional_import(module_name: str) -> Any | None:
    try:
        return import_module(module_name)
    except ImportError:
        return None
