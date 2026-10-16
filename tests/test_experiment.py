from fal.facade import FAL
from fal.result import FALResult


class FakeExperiment:
    def execute(self):
        return FALResult(model=object(), timings={"rounds": []})


class FakeBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        return FakeExperiment()


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
