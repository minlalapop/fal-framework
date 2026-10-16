import csv
import json
import os

from experiments import ExperimentSpec, run_campaign
from fal.result import FALResult


class FakeConfig:
    dataset = "synthetic"
    model = object()
    X = None
    y = None


class FakeFAL:
    def __init__(self, seed):
        self.seed = seed
        self.config = FakeConfig()

    def run(self):
        assert str(os.environ.get("PYTHONPATH", ""))
        score = self.seed / 100.0
        return FALResult(
            model=object(),
            losses=[(1, 1.0 - score)],
            metrics={"accuracy": [(1, score)]},
            timings={
                "total_execution_time": float(self.seed),
                "rounds": [{"round": 1, "round_execution_time": 0.5}],
                "clients": [],
            },
        )


def test_runner_consolidates_runs_rounds_and_summary(tmp_path):
    previous_pythonpath = os.environ.get("PYTHONPATH")
    result = run_campaign(
        [
            ExperimentSpec(
                name="Synthetic Clean",
                build_fal=FakeFAL,
                seeds=(1, 3),
                metadata={"training": "clean"},
            )
        ],
        output_dir=tmp_path / "campaign",
    )

    campaign_dir = tmp_path / "campaign"
    with (campaign_dir / "runs.csv").open(newline="", encoding="utf-8") as file:
        runs = list(csv.DictReader(file))
    with (campaign_dir / "round_metrics.csv").open(newline="", encoding="utf-8") as file:
        rounds = list(csv.DictReader(file))
    with (campaign_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    with (campaign_dir / "campaign.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert result["directory"] == str(campaign_dir.resolve())
    assert [row["seed"] for row in runs] == ["1", "3"]
    assert [row["metric_accuracy"] for row in rounds] == ["0.01", "0.03"]
    accuracy_summary = next(row for row in summary if row["measure"] == "metric_accuracy")
    assert float(accuracy_summary["mean"]) == 0.02
    assert manifest["successful_runs"] == 2
    assert (campaign_dir / "synthetic_clean" / "seed_1" / "run.json").exists()
    assert os.environ.get("PYTHONPATH") == previous_pythonpath
