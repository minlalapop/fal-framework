from fal.facade import FAL
from fal.experiment import Experiment
from fal.result import FALResult


class FakeExperiment:
    def execute(self):
        return FALResult(model=object(), timings={"rounds": []})


class FakeBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        return FakeExperiment()


class PickleableModel:
    pass


def test_fal_records_total_execution_time(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("fal.facade.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr("fal.facade.CoreBuilder", FakeBuilder)
    fal = FAL(
        dataset="iris",
        model=object(),
        verbose=0,
    )

    result = fal.run()

    assert result.timings["total_execution_time"] == 2.5


def test_experiment_exports_its_configuration_with_other_results(tmp_path, monkeypatch):
    class DatasetManager:
        def prepare(self):
            return self

        def get_clients_data(self):
            return [([1], [0])]

        def get_test_data(self):
            return ([2], [0])

    class FLSystem:
        def start(self, **kwargs):
            return FALResult(model=PickleableModel())

    monkeypatch.chdir(tmp_path)
    experiment = Experiment(
        dataset_manager=DatasetManager(),
        model=object(),
        trainer=object(),
        metrics_manager=object(),
        fl_system=FLSystem(),
        config={"dataset": "iris", "random_state": 42},
        verbose=0,
    )

    result = experiment.execute()

    assert result.config == {"dataset": "iris", "random_state": 42}
    assert "config" in result.artifact_paths
