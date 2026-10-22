"""Attack backend factory."""

from __future__ import annotations

from typing import Any

from fal.adv.art.art_backend import ARTBackend
from fal.adv.backend_base import AttackBackend


class AttackBackendFactory:
    """Create an attack backend."""

    _registry = {
        "art": ARTBackend,
    }

    @classmethod
    def register(cls, name: str, backend_cls: type[AttackBackend]) -> None:
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, AttackBackend):
            raise TypeError("Registered attack backends must inherit from AttackBackend.")
        cls._registry[cls._normalize_name(name)] = backend_cls

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

    @classmethod
    def create(
        cls,
        backend_name: str = "art",
        backend_params: dict[str, Any] | None = None,
    ) -> AttackBackend:
        backend_name = cls._normalize_name(backend_name)
        backend_cls = cls._registry.get(backend_name)

        if backend_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported attack backend: {backend_name}. Supported backends: {supported}.")

        return backend_cls(**(backend_params or {}))

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Backend name must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
