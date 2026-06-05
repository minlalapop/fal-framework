import csv
import json
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
