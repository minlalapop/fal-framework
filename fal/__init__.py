"""Federated Adversarial Learning framework."""

from fal.config import FALConfig
from fal.facade import FAL
from fal.result import FALResult

__all__ = [
    "FAL",
    "FALConfig",
    "FALResult",
]
