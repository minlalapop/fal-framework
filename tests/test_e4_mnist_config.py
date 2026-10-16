from experiments.configs.e1_e2_mnist import build_experiments as build_iid
from experiments.configs.e4_mnist_non_iid import build_experiments as build_non_iid


def test_e4_changes_only_partition_configuration():
    iid = build_iid(smoke=False)[0].build_fal(42).config
    non_iid = build_non_iid(smoke=False)[0].build_fal(42).config

    assert iid.iid is True
    assert non_iid.iid is False
    assert non_iid.shards_per_client == 2
    assert type(iid.model) is type(non_iid.model)
    assert iid.trainer_type == non_iid.trainer_type == "clean"
    assert iid.num_clients == non_iid.num_clients == 3
    assert iid.num_rounds == non_iid.num_rounds == 5
    assert iid.model_params["epochs"] == non_iid.model_params["epochs"] == 5
    assert iid.random_state == non_iid.random_state == 42
