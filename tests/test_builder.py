import pytest
from sklearn.linear_model import LogisticRegression

from fal.builder import CoreBuilder
from fal.config import FALConfig


def test_builder_passes_non_iid_splitter_options_from_config():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        iid=False,
        shards_per_client=3,
        shuffle=False,
        random_state=42,
    )

    splitter = CoreBuilder(config)._build_client_splitter()

    assert splitter.shards_per_client == 3
    assert splitter.shuffle is False
    assert splitter.random_state == 42


def test_builder_rejects_sklearn_fedprox_before_flower_execution():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        fl_algorithm="fedprox",
        fl_params={"proximal_mu": 0.1},
    )

    with pytest.raises(ValueError, match="FedProx proximal training is not supported for sklearn"):
        CoreBuilder(config).build()


def test_attack_only_sklearn_defaults_to_clean_fit_when_unfitted():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        trainer_type="attack_only",
        attack_type="fgsm",
    )
    builder = CoreBuilder(config)
    model = builder._build_model()

    trainer = builder._build_trainer(
        model=model,
        metrics_manager=None,
        loss_manager=None,
        attack=object(),
    )

    assert trainer.trainer_params["clean_fit"] is True


def test_attack_only_respects_explicit_clean_fit_false():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        trainer_type="attack_only",
        trainer_params={"clean_fit": False},
        attack_type="fgsm",
    )
    builder = CoreBuilder(config)
    model = builder._build_model()

    trainer = builder._build_trainer(
        model=model,
        metrics_manager=None,
        loss_manager=None,
        attack=object(),
    )

    assert trainer.trainer_params["clean_fit"] is False
