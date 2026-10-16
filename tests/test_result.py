import csv
import json
import math
from pathlib import Path

from fal.result import FALResult


class PickleableModel:
    pass


def test_result_export_creates_incremental_run_directories(tmp_path):
    base_dir = tmp_path / "results"
    first = FALResult(
        model=PickleableModel(),
        losses=[(1, 0.8)],
        metrics={"accuracy": [(1, 0.9)]},
    )
    second = FALResult(
        model=PickleableModel(),
        losses=[(1, 0.7)],
        metrics={"accuracy": [(1, 0.95)]},
    )

    first_paths = first.save_outputs(str(base_dir))
    second_paths = second.save_outputs(str(base_dir), save_model=False)

    assert Path(first_paths["directory"]).name == "results_1"
    assert Path(second_paths["directory"]).name == "results_2"
    assert (base_dir / "results_1" / "model.pkl").exists()
    assert not (base_dir / "results_2" / "model.pkl").exists()


def test_result_export_writes_scalar_and_detailed_metrics_csv(tmp_path):
    result = FALResult(
        model=PickleableModel(),
        losses=[(1, 0.8), (2, 0.6)],
        metrics={
            "accuracy": [(1, 0.7), (2, 0.9)],
            "classification_report": [
                (2, {"0": {"precision": 1.0, "recall": 0.5, "f1-score": 0.67, "support": 2}})
            ],
            "confusion_matrix": [(2, [[1, 1], [0, 2]])],
        },
    )

    paths = result.save_outputs(str(tmp_path / "results"))

    with open(paths["metrics"], newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0] == {"round": "1", "loss": "0.8", "accuracy": "0.7"}
    assert rows[1] == {"round": "2", "loss": "0.6", "accuracy": "0.9"}

    with open(paths["detailed_metrics"], newline="", encoding="utf-8") as file:
        detailed_rows = list(csv.DictReader(file))

    assert detailed_rows[0]["round"] == "2"
    assert json.loads(detailed_rows[0]["confusion_matrix"]) == [[1, 1], [0, 2]]
    assert json.loads(detailed_rows[0]["classification_report"])["0"]["precision"] == 1.0


def test_result_discards_unavailable_losses_and_exports_timings(tmp_path):
    result = FALResult(
        model=PickleableModel(),
        losses=[(1, math.nan), (2, 0.4)],
        metrics={"accuracy": [(1, 0.7), (2, 0.8)]},
        timings={
            "total_execution_time": 2.5,
            "rounds": [{"round": 1, "round_execution_time": 1.0}],
        },
    )

    paths = result.save_outputs(str(tmp_path / "results"), save_model=False)

    assert result.losses == [(2, 0.4)]
    assert result.loss_available is True
    with open(paths["metrics"], newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["loss"] == ""
    assert rows[1]["loss"] == "0.4"
    with open(paths["timings"], encoding="utf-8") as file:
        timings = json.load(file)
    assert timings["total_execution_time"] == 2.5

    result.timings["total_execution_time"] = 3.0
    result.refresh_timings_output()
    with open(paths["timings"], encoding="utf-8") as file:
        refreshed_timings = json.load(file)
    assert refreshed_timings["total_execution_time"] == 3.0


def test_result_without_real_loss_has_no_loss_column(tmp_path):
    result = FALResult(
        model=PickleableModel(),
        losses=[(1, math.nan)],
        metrics={"accuracy": [(1, 0.7)]},
    )

    paths = result.save_outputs(str(tmp_path / "results"), save_model=False)

    assert result.losses == []
    assert result.latest_loss is None
    assert result.loss_available is False
    with open(paths["metrics"], newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert "loss" not in rows[0]
