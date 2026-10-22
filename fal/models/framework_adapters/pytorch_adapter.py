"""Adapter for PyTorch modules."""

from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

import numpy as np

from fal.models.base import BaseModel


class PyTorchModelAdapter(BaseModel):
    """Wrap a ``torch.nn.Module`` behind the ``BaseModel`` interface."""

    def __init__(self, model: Any, **model_params: Any) -> None:
        self.model = model
        self.criterion = model_params.get("criterion")
        self.optimizer = model_params.get("optimizer")
        self.epochs = int(model_params.get("epochs", 1))
        self.batch_size = int(model_params.get("batch_size", 32))
        self.predict_batch_size = int(model_params.get("predict_batch_size", self.batch_size))
        self.device = model_params.get("device")

        self._torch = _require_torch()
        self._local_training_reference: list[Any] | None = None
        self._validate_model()
        self._configure_device()
        self._validate_training_params()

    @property
    def framework(self) -> str:
        return "pytorch"

    def fit(self, X: Any, y: Any, **params: Any) -> "PyTorchModelAdapter":
        self.configure_runtime_device()
        epochs = int(params.get("local_steps", params.get("epochs", self.epochs)))
        batch_size = int(params.get("batch_size", self.batch_size))
        proximal_mu = float(params.get("proximal_mu", 0.0) or 0.0)
        global_params = self._local_training_reference
        if proximal_mu > 0 and global_params is None:
            global_params = self._snapshot_parameters()

        dataset = self._make_tensor_dataset(X, y)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss = self._add_proximal_term(loss, global_params, proximal_mu)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X: Any) -> Any:
        self.configure_runtime_device()
        tensor_X = self._to_tensor(X)
        loader = self._torch.utils.data.DataLoader(
            self._torch.utils.data.TensorDataset(tensor_X),
            batch_size=self.predict_batch_size,
            shuffle=False,
        )

        self.model.eval()
        predictions = []
        with self._torch.no_grad():
            for (batch_X,) in loader:
                outputs = self.model(batch_X.to(self.device))

                if outputs.ndim > 1 and outputs.shape[-1] > 1:
                    batch_predictions = outputs.argmax(dim=-1)
                else:
                    batch_predictions = (outputs.reshape(-1) >= 0.5).long()

                predictions.append(batch_predictions.detach().cpu())

        return self._torch.cat(predictions).numpy()

    def get_raw_model(self) -> Any:
        return self.model

    def configure_runtime_device(self) -> "PyTorchModelAdapter":
        """Move model, criterion, and optimizer state to the requested device."""

        self._validate_device_available(self.device)
        self.model.to(self.device)
        if callable(getattr(self.criterion, "to", None)):
            self.criterion.to(self.device)
        self._move_optimizer_state(self.device)
        return self

    def _begin_local_training(self, params: dict[str, Any] | None = None) -> None:
        proximal_mu = float((params or {}).get("proximal_mu", 0.0) or 0.0)
        self._local_training_reference = (
            self._snapshot_parameters() if proximal_mu > 0 else None
        )

    def _end_local_training(self) -> None:
        self._local_training_reference = None

    def clone(self) -> "PyTorchModelAdapter":
        cloned_model = copy.deepcopy(self.model)
        cloned_criterion = copy.deepcopy(self.criterion)
        cloned_optimizer = self._clone_optimizer(cloned_model)

        return type(self)(
            cloned_model,
            criterion=cloned_criterion,
            optimizer=cloned_optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            predict_batch_size=self.predict_batch_size,
            device=str(self.device),
        )

    def _validate_model(self) -> None:
        module_cls = self._torch.nn.Module
        if not isinstance(self.model, module_cls):
            raise TypeError("A PyTorch model must be an instance of `torch.nn.Module`.")

    def _configure_device(self) -> None:
        if self.device is None:
            self.device = "cpu"
        try:
            self.device = self._torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Unsupported PyTorch device `{self.device}`.") from exc
        self._validate_device_available(self.device)

        # Keep serialised client templates on CPU. Flower/Ray workers move their
        # own copy to the requested runtime device after Ray assigns accelerators.
        self.model.to("cpu")
        if callable(getattr(self.criterion, "to", None)):
            self.criterion.to("cpu")
        self._move_optimizer_state(self._torch.device("cpu"))

    def _validate_device_available(self, device: Any) -> None:
        if device.type == "cpu":
            return

        if device.type == "cuda":
            if not self._torch.cuda.is_available():
                raise ValueError(
                    "PyTorch device `cuda` was requested, but CUDA is not available. "
                    "Use `device='cpu'` or run on a CUDA-enabled host."
                )
            if device.index is not None and device.index >= self._torch.cuda.device_count():
                raise ValueError(
                    f"PyTorch device `{device}` does not exist; "
                    f"CUDA exposes {self._torch.cuda.device_count()} device(s)."
                )
            return

        if device.type == "mps":
            mps = getattr(self._torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ValueError(
                    "PyTorch device `mps` was requested, but MPS is not available."
                )
            return

        raise ValueError(
            f"Unsupported PyTorch device `{device}`. FAL supports `cpu`, `cuda`, "
            "and validates `mps` separately."
        )

    def _move_optimizer_state(self, device: Any) -> None:
        state = getattr(self.optimizer, "state", None)
        if not isinstance(state, dict):
            return
        for parameter_state in state.values():
            if not isinstance(parameter_state, dict):
                continue
            for name, value in parameter_state.items():
                parameter_state[name] = self._move_value_to_device(value, device)

    def _move_value_to_device(self, value: Any, device: Any) -> Any:
        if self._torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {
                name: self._move_value_to_device(item, device)
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self._move_value_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_value_to_device(item, device) for item in value)
        return value

    def _validate_training_params(self) -> None:
        if self.criterion is None:
            raise ValueError("PyTorch model_params must include `criterion`.")
        if self.optimizer is None:
            raise ValueError("PyTorch model_params must include `optimizer`.")
        if self.epochs <= 0:
            raise ValueError("`epochs` must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if self.predict_batch_size <= 0:
            raise ValueError("`predict_batch_size` must be a positive integer.")

    def _make_tensor_dataset(self, X: Any, y: Any) -> Any:
        tensor_X = self._to_tensor(X)
        tensor_y = self._to_target_tensor(y)
        return self._torch.utils.data.TensorDataset(tensor_X, tensor_y)

    def _snapshot_parameters(self) -> list[Any]:
        return [parameter.detach().clone() for parameter in self.model.parameters()]

    def _add_proximal_term(
        self,
        loss: Any,
        global_params: list[Any] | None,
        proximal_mu: float,
    ) -> Any:
        if not global_params or proximal_mu <= 0:
            return loss

        proximal_term = 0.0
        for local_parameter, global_parameter in zip(self.model.parameters(), global_params):
            difference = local_parameter - global_parameter.to(local_parameter.device)
            proximal_term = proximal_term + self._torch.sum(difference.pow(2))

        return loss + (proximal_mu / 2.0) * proximal_term

    def _to_tensor(self, values: Any) -> Any:
        if self._torch.is_tensor(values):
            tensor = values
        else:
            if isinstance(values, np.ndarray) and not values.flags.writeable:
                values = values.copy()
            tensor = self._torch.as_tensor(values)

        if not self._torch.is_floating_point(tensor):
            tensor = tensor.float()
        return tensor

    def _to_target_tensor(self, values: Any) -> Any:
        if self._torch.is_tensor(values):
            tensor = values
        else:
            if isinstance(values, np.ndarray) and not values.flags.writeable:
                values = values.copy()
            tensor = self._torch.as_tensor(values)

        return tensor.long()

    def _clone_optimizer(self, cloned_model: Any) -> Any:
        optimizer_cls = type(self.optimizer)
        defaults = dict(getattr(self.optimizer, "defaults", {}))
        try:
            return optimizer_cls(cloned_model.parameters(), **defaults)
        except TypeError:
            return optimizer_cls(cloned_model.parameters())


def _require_torch() -> Any:
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "PyTorch support requires `torch`. Install it before using PyTorch models."
        ) from exc
    return torch
