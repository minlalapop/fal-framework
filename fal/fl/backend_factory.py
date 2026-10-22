
"""Factory for federated learning backends."""

from __future__ import annotations

from fal.fl.backend_base import FLBackend
from fal.fl.flower.flower_backend import FlowerBackend


class FLBackendFactory:
    """Create FL backends."""

    _registry = {
        "flower": FlowerBackend,
    }

    @classmethod
    def register(cls, name: str, backend_cls: type[FLBackend]) -> None:
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, FLBackend):
            raise TypeError("Registered FL backends must inherit from FLBackend.")
        cls._registry[cls._normalize_name(name)] = backend_cls

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

    @classmethod
    def create(cls, backend_name: str = "flower") -> FLBackend:
        backend_name = cls._normalize_name(backend_name)
        backend_cls = cls._registry.get(backend_name)

        if backend_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported FL backend: {backend_name}. Supported: {supported}.")

        return backend_cls()

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("FL backend name must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
