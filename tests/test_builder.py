import pickle

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from fal import FAL
from fal.adv import AttackBackend, AttackBackendFactory, AttackFactory, BaseAttack
from fal.builder import CoreBuilder
from fal.config import FALConfig
from fal.dataset import BuiltinDatasetLoader, DatasetLoader
from fal.evaluation.losses import BaseEvaluationLoss, LossFactory
from fal.evaluation.metrics import BaseMetric, MetricFactory
from fal.fl import FLBackend, FLBackendFactory
from fal.fl.flower.flower_strategy import StrategyAdapter, StrategyFactory
from fal.models import BaseModel, ModelAdapterFactory
from fal.result import FALResult
from fal.trainers import BaseTrainer, TrainerFactory


class CustomRawModel:
    def __init__(self):
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


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


def test_builder_attaches_original_config_to_experiment_exports():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        random_state=42,
    )

    experiment = CoreBuilder(config).build()

    assert experiment.config is config


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


def test_builder_passes_art_estimator_configuration_to_attack_backend():
    bounds = (
        np.array([-2.0, 10.0]),
        np.array([2.0, 20.0]),
    )
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        trainer_type="attack_only",
        attack_type="fgsm",
        attack_backend_params={"estimator_config": {"clip_values": bounds}},
    )

    attack = CoreBuilder(config)._build_attack()

    actual = attack.backend.estimator_config["clip_values"]
    assert np.array_equal(actual[0], bounds[0])
    assert np.array_equal(actual[1], bounds[1])


def test_builder_propagates_runtime_resources_outside_strategy_params():
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        num_cpus=2,
        num_gpus=0,
        max_concurrent_clients=1,
        fl_params={"fraction_fit": 1.0},
    )
    builder = CoreBuilder(config)

    run_config = builder._build_run_config()
    strategy = builder._build_fl_system().strategy

    assert run_config["num_cpus"] == 2.0
    assert run_config["num_gpus"] == 0.0
    assert run_config["max_concurrent_clients"] == 1
    assert strategy.fl_params == {"fraction_fit": 1.0}


@pytest.mark.parametrize(
    ("algorithm", "fl_params"),
    [
        ("fedavg", {}),
        ("fedprox", {"proximal_mu": 0.1}),
        ("fedadam", {}),
    ],
)
def test_runtime_resources_are_strategy_independent(algorithm, fl_params):
    config = FALConfig(
        dataset="iris",
        model=LogisticRegression(),
        fl_algorithm=algorithm,
        fl_params=fl_params,
        num_cpus=1,
        num_gpus=0,
        max_concurrent_clients=1,
    )

    fl_system = CoreBuilder(config)._build_fl_system()

    assert fl_system.run_config["num_cpus"] == 1.0
    assert fl_system.run_config["num_gpus"] == 0.0
    assert fl_system.run_config["max_concurrent_clients"] == 1
    assert fl_system.strategy.fl_params == fl_params


def test_registered_components_run_through_public_fal_api(tmp_path, monkeypatch, request):
    calls = set()

    class CustomDatasetLoader(DatasetLoader):
        def __init__(self, offset=0):
            self.offset = offset

        def load(self):
            calls.add("dataset")
            X = np.arange(48, dtype=np.float32).reshape(24, 2) + self.offset
            y = np.tile(np.array([0, 1, 2]), 8)
            return X, y

    class CustomModelAdapter(BaseModel):
        def __init__(self, model, **params):
            calls.add("model_adapter")
            self.model = model

        @property
        def framework(self):
            return "custom_model"

        def fit(self, X, y, **params):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def get_raw_model(self):
            return self.model

    class CustomAttackBackend(AttackBackend):
        def generate(self, attack_name, model, X, y, params):
            calls.add("attack_backend")
            return X

    class CustomAttack(BaseAttack):
        def generate(self, model, X, y=None):
            calls.add("attack")
            return self.backend.generate("custom_attack", model, X, y, self.params)

    class CustomMetric(BaseMetric):
        name = "custom_metric"

        def compute(self, y_true, y_pred):
            calls.add("metric")
            return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    class CustomLoss(BaseEvaluationLoss):
        name = "custom_loss"

        def compute(self, model, X, y_true):
            calls.add("loss")
            return 0.25

    class CustomTrainer(BaseTrainer):
        def train(self, data):
            calls.add("trainer")
            X, y = data
            self.attack.generate(self.model, X, y)
            self.model.fit(X, y)

    class CustomStrategy(StrategyAdapter):
        def build_strategy(self):
            calls.add("strategy")
            return "custom_strategy"

    class CustomFLBackend(FLBackend):
        def start(
            self,
            model,
            trainer,
            clients_data,
            test_data,
            strategy,
            num_rounds=1,
            verbose=1,
            run_config=None,
        ):
            calls.add("fl_backend")
            assert strategy.build_strategy() == "custom_strategy"
            trainer.train(clients_data[0])
            evaluation = trainer.evaluate(test_data)
            return FALResult(
                model=model,
                losses=[(1, evaluation.loss)],
                metrics={
                    name: [(1, value)]
                    for name, value in evaluation.metrics.items()
                },
                backend="custom_fl_backend",
            )

    registrations = (
        (AttackBackendFactory, "custom_backend", CustomAttackBackend),
        (AttackFactory, "custom_attack", CustomAttack),
        (MetricFactory, "custom_metric", CustomMetric),
        (LossFactory, "custom_loss", CustomLoss),
        (TrainerFactory, "custom_trainer", CustomTrainer),
        (StrategyFactory, "custom_strategy", CustomStrategy),
        (FLBackendFactory, "custom_fl_backend", CustomFLBackend),
    )
    for factory, name, component in registrations:
        factory.register(name, component)
        request.addfinalizer(lambda factory=factory, name=name: factory._registry.pop(name, None))

    BuiltinDatasetLoader.register("custom_dataset", CustomDatasetLoader)
    request.addfinalizer(
        lambda: BuiltinDatasetLoader.SUPPORTED_DATASETS.pop("custom_dataset", None)
    )
    ModelAdapterFactory.register(
        "custom_model",
        CustomModelAdapter,
        detector=lambda model: isinstance(model, CustomRawModel),
    )
    request.addfinalizer(lambda: ModelAdapterFactory._registry.pop("custom_model", None))
    request.addfinalizer(lambda: ModelAdapterFactory._detectors.pop("custom_model", None))

    monkeypatch.chdir(tmp_path)
    result = FAL(
        dataset="custom_dataset",
        dataset_params={"offset": 1},
        model=CustomRawModel(),
        trainer_type="custom_trainer",
        attack_type="custom_attack",
        attack_backend="custom_backend",
        metrics=["custom_metric"],
        loss="custom_loss",
        fl_backend="custom_fl_backend",
        fl_algorithm="custom_strategy",
        num_clients=2,
        num_rounds=1,
        random_state=42,
        verbose=0,
    ).run()

    assert calls == {
        "attack_backend",
        "attack",
        "dataset",
        "metric",
        "model_adapter",
        "loss",
        "trainer",
        "strategy",
        "fl_backend",
    }
    assert result.backend == "custom_fl_backend"
    assert result.losses == [(1, 0.25)]
    assert result.metrics["custom_metric"][0][0] == 1
    assert "config" in result.artifact_paths
    assert result.raw_model.was_fit is True
    with open(result.artifact_paths["model"], "rb") as saved_model:
        restored = pickle.load(saved_model)
    assert isinstance(restored, CustomRawModel)
    assert restored.was_fit is True


def test_multiple_registered_components_do_not_conflict(request):
    class FirstMetric(BaseMetric):
        name = "first_metric"

        def compute(self, y_true, y_pred):
            return 1.0

    class SecondMetric(BaseMetric):
        name = "second_metric"

        def compute(self, y_true, y_pred):
            return 2.0

    MetricFactory.register("first_metric", FirstMetric)
    MetricFactory.register("second_metric", SecondMetric)
    request.addfinalizer(lambda: MetricFactory._registry.pop("first_metric", None))
    request.addfinalizer(lambda: MetricFactory._registry.pop("second_metric", None))

    manager = MetricFactory.create(["first_metric", "second_metric"])

    assert manager.compute_all([0], [0]) == {
        "first_metric": 1.0,
        "second_metric": 2.0,
    }


def test_registration_rejects_component_with_wrong_base_class():
    with pytest.raises(TypeError, match="BaseTrainer"):
        TrainerFactory.register("not_a_trainer", object)
