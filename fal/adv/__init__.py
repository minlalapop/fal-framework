"""Attack abstractions exposed by the FAL framework."""

from fal.adv.art.art_model_adapter import ARTModelAdapter
from fal.adv.art.art_backend import ARTBackend
from fal.adv.backend_factory import AttackBackendFactory
from fal.adv.attacks.attack_factory import AttackFactory
from fal.adv.backend_base import AttackBackend
from fal.adv.attacks.base import BaseAttack
from fal.adv.attacks.evasion.deepfool import DeepFool
from fal.adv.attacks.evasion.fgsm import FGSM
from fal.adv.attacks.evasion.pgd import PGD
from fal.adv.art.pytorch_art_adapter import PyTorchARTAdapter
from fal.adv.art.sklearn_art_adapter import SklearnARTAdapter

__all__ = [
    "ARTBackend",
    "ARTModelAdapter",
    "AttackBackend",
    "AttackBackendFactory",
    "AttackFactory",
    "BaseAttack",
    "DeepFool",
    "FGSM",
    "PGD",
    "PyTorchARTAdapter",
    "SklearnARTAdapter",
]
