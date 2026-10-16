import random

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fal import seed_everything
from fal.config import FALConfig


def test_config_accepts_comparison_trainer_alias():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        trainer_type=" Comparison ",
        attack_type="FGSM",
    )

    config.validate()

    assert config.trainer_type == "comparison"
    assert config.attack_type == "fgsm"


def test_adversarial_trainers_require_attack_type():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        trainer_type="adversarial",
    )

    with pytest.raises(ValueError, match="requires `attack_type`"):
        config.validate()


def test_unknown_metric_fails_with_supported_names():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        metrics=["accuracy", "unknown_metric"],
    )

    with pytest.raises(ValueError, match="Unsupported metrics"):
        config.validate()


def test_non_iid_shards_per_client_must_be_positive():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        iid=False,
        shards_per_client=0,
    )

    with pytest.raises(ValueError, match="shards_per_client"):
        config.validate()


def test_config_accepts_deterministic_seed_control():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        random_state=42,
        deterministic=True,
    )

    config.validate()

    assert config.random_state == 42
    assert config.deterministic is True


def test_config_rejects_non_boolean_deterministic_option():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        deterministic="yes",
    )

    with pytest.raises(ValueError, match="deterministic"):
        config.validate()


def test_public_seed_helper_controls_python_and_numpy_rngs():
    seed_everything(42)
    first = (random.random(), np.random.random())
    seed_everything(42)
    second = (random.random(), np.random.random())

    assert first == second
