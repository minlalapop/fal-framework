"""PyTorch Flower parameter adapter."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from fal.fl.flower.flower_client.flower_parameters.flower_parameters_adapter import (
    FlowerParameterAdapter,
)


class PyTorchParameterAdapter(FlowerParameterAdapter):
    def supports(self, model: Any) -> bool:
        if getattr(model, "framework", None) != "pytorch":
            return False
        raw = model.get_raw_model()
        return hasattr(raw, "state_dict") and hasattr(raw, "load_state_dict")

    def initialize(self, model: Any, X: Any, y: Any, classes: Any | None = None) -> None:
        return None

    def get_parameters(self, model: Any) -> list:
        raw = model.get_raw_model()
        return [
            value.detach().cpu().numpy()
            for value in raw.state_dict().values()
        ]

    def set_parameters(self, model: Any, parameters: list) -> None:
        raw = model.get_raw_model()
        keys = list(raw.state_dict().keys())

        state_dict = OrderedDict(
            {
                key: torch.tensor(value)
                for key, value in zip(keys, parameters)
            }
        )

        raw.load_state_dict(state_dict, strict=True)
