"""Adversarial attack implementations."""

from fal.adv.attacks.attack_factory import AttackFactory
from fal.adv.attacks.base import BaseAttack
from fal.adv.attacks.evasion.deepfool import DeepFool
from fal.adv.attacks.evasion.fgsm import FGSM
from fal.adv.attacks.evasion.pgd import PGD

__all__ = [
    "AttackFactory",
    "BaseAttack",
    "DeepFool",
    "FGSM",
    "PGD",
]
