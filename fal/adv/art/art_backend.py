"""ART attack backend implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fal.adv.art.art_model_adapter import ARTModelAdapter, validate_clip_values
from fal.adv.backend_base import AttackBackend
from fal.adv.art.pytorch_art_adapter import PyTorchARTAdapter
from fal.adv.art.sklearn_art_adapter import SklearnARTAdapter


class ARTBackend(AttackBackend):
    """Backend that wraps ART attack generation."""

    def __init__(
        self,
        art_model_adapter: ARTModelAdapter | None = None,
        estimator_config: dict[str, Any] | None = None,
    ) -> None:
        if estimator_config is not None and not isinstance(estimator_config, dict):
            raise ValueError("ART `estimator_config` must be a dictionary.")
        self.FastGradientMethod, self.ProjectedGradientDescent, self.DeepFool = _require_art_attacks()
        self.art_model_adapter = art_model_adapter
        self.estimator_config = dict(estimator_config or {})
        if "clip_values" in self.estimator_config:
            self.estimator_config["clip_values"] = validate_clip_values(
                self.estimator_config["clip_values"]
            )

    def generate(
        self,
        attack_name: str,
        model: Any,
        X: Any,
        y: Any | None,
        params: dict[str, Any],
    ) -> Any:
        estimator = self.build_estimator(model, X=X, y=y)

        if attack_name == "fgsm":
            attack = self.FastGradientMethod(estimator=estimator, **params)
        elif attack_name == "pgd":
            attack = self.ProjectedGradientDescent(estimator=estimator, **params)
        elif attack_name == "deepfool":
            attack = self.DeepFool(classifier=estimator, **params)
        else:
            raise ValueError("Unsupported ART attack. Supported attacks: fgsm, pgd, deepfool.")

        if y is None:
            return attack.generate(x=X)
        return attack.generate(x=X, y=y)

    def build_estimator(self, model: Any, X: Any | None = None, y: Any | None = None) -> Any:
        if self.art_model_adapter is not None:
            return self.art_model_adapter.create_estimator(model, self.estimator_config, X=X, y=y)

        if model.framework == "pytorch":
            adapter = PyTorchARTAdapter()
        elif model.framework == "sklearn":
            adapter = SklearnARTAdapter()
        else:
            raise ValueError(f"Unsupported model framework `{model.framework}` for ART.")

        return adapter.create_estimator(model, self.estimator_config, X=X, y=y)


def _require_art_attacks() -> tuple[Any, Any, Any]:
    try:
        evasion = import_module("art.attacks.evasion")
    except ImportError as exc:
        raise ImportError("ARTBackend requires `adversarial-robustness-toolbox`.") from exc
    return evasion.FastGradientMethod, evasion.ProjectedGradientDescent, evasion.DeepFool
