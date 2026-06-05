from .clean import CleanTrainer
from .attack_only import AttackOnlyTrainer
from .adversarial import AdversarialTrainer
from .mixed import MixedTrainer


class TrainerFactory:
    _registry = {
        "adversarial": AdversarialTrainer,
        "attack_only": AttackOnlyTrainer,
        "clean": CleanTrainer,
        "comparison": MixedTrainer,
        "mixed": MixedTrainer,
    }

    @classmethod
    def register(cls, name, trainer_cls):
        cls._registry[cls._normalize_name(name)] = trainer_cls

    @classmethod
    def create(
        cls,
        trainer_type,
        model,
        attack=None,
        metrics_manager=None,
        loss_manager=None,
        trainer_params=None,
    ):
        trainer_type = cls._normalize_name(trainer_type)
        trainer_params = trainer_params or {}
        trainer_cls = cls._registry.get(trainer_type)

        if trainer_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported trainer_type: {trainer_type}. Supported: {supported}.")

        return trainer_cls(
            model=model,
            attack=attack,
            metrics_manager=metrics_manager,
            loss_manager=loss_manager,
            trainer_params=trainer_params,
        )

    @staticmethod
    def _normalize_name(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("trainer_type must be a non-empty string.")
        return value.strip().lower().replace("-", "_")
