"""Compatibility exports for attack classes."""

from fal.adv.attacks.evasion.deepfool import DeepFool
from fal.adv.attacks.evasion.fgsm import FGSM
from fal.adv.attacks.evasion.pgd import PGD

__all__ = [
    "DeepFool",
    "FGSM",
    "PGD",
]
