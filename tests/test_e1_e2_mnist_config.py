from experiments.configs.e1_e2_mnist import SEEDS, build_experiments


def test_mnist_campaign_defines_controlled_training_modes():
    specs = build_experiments(smoke=False)

    assert [spec.metadata["training"] for spec in specs] == [
        "clean",
        "attack_only",
        "adversarial",
    ]
    assert all(tuple(spec.seeds) == SEEDS for spec in specs)
    assert all(spec.deterministic for spec in specs)

    configs = [spec.build_fal(42).config for spec in specs]
    assert [config.attack_type for config in configs] == [None, "pgd", "pgd"]
    assert all(config.iid for config in configs)
    assert all(config.num_clients == 3 for config in configs)
    assert all(config.num_rounds == 5 for config in configs)
    assert all(config.random_state == 42 for config in configs)
    assert all(config.loss == "model_loss" for config in configs)


def test_mnist_smoke_campaign_is_explicitly_reduced():
    specs = build_experiments(smoke=True)
    config = specs[0].build_fal(42).config

    assert tuple(specs[0].seeds) == (42,)
    assert specs[0].metadata["smoke"] is True
    assert config.num_rounds == 1
    assert config.model_params["epochs"] == 1
    assert config.dataset_params["samples_per_class"] == 30
