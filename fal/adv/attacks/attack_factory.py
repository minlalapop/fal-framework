"""Attack factory."""

from __future__ import annotations

from typing import Any

from fal.adv.backend_base import AttackBackend
from fal.adv.attacks.base import BaseAttack
from fal.adv.attacks.evasion.deepfool import DeepFool
from fal.adv.attacks.evasion.fgsm import FGSM
from fal.adv.attacks.evasion.pgd import PGD


class AttackFactory:
    """Create a concrete attack object."""

    _registry = {
        "deepfool": DeepFool,
        "fgsm": FGSM,
        "pgd": PGD,
    }

    @classmethod
    def register(cls, name: str, attack_cls: type[BaseAttack]) -> None:
        if not isinstance(attack_cls, type) or not issubclass(attack_cls, BaseAttack):
            raise TypeError("Registered attacks must inherit from BaseAttack.")
        cls._registry[cls._normalize_name(name)] = attack_cls

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

    @classmethod
    def create(
        cls,
        attack_type: str,
        attack_params: dict[str, Any] | None,
        backend: AttackBackend,
    ) -> BaseAttack:
        attack_type = cls._normalize_name(attack_type)
        params = attack_params or {}
        attack_cls = cls._registry.get(attack_type)

        if attack_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported attack: {attack_type}. Supported attacks: {supported}.")

        return attack_cls(backend, params)

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Attack type must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
