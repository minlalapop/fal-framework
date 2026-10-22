import csv
import json
import math
from pathlib import Path

import numpy as np

from fal.progress import _format_header, _ordered_metric_names as _ordered_progress_metrics
from fal.result import FALResult


class PickleableModel:
    pass


class FakeOptimizer:
    def __init__(self):
        self.defaults = {"lr": 0.01, "momentum": 0.9}
        self.param_groups = [{"params": [object()], "lr": 0.01}]


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


def test_result_exports_safe_reproducibility_config(tmp_path):
    result = FALResult(
        model=PickleableModel(),
        config={
            "dataset": "digits",
            "model": PickleableModel(),
            "model_params": {
                "optimizer": FakeOptimizer(),
                "api_token": "must-not-be-exported",
            },
            "trainer_type": "mixed",
            "random_state": 42,
            "X": np.zeros((8, 4), dtype=np.float32),
            "y": np.zeros(8, dtype=np.int64),
        },
    )

    paths = result.save_outputs(str(tmp_path / "results"), save_model=False)

    with open(paths["config"], encoding="utf-8") as file:
        config = json.load(file)

    assert config["dataset"] == "array"
    assert config["array_data"]["X"]["shape"] == [8, 4]
    assert config["array_data"]["X"]["dtype"] == "float32"
    assert config["model"]["type"].endswith("PickleableModel")
    assert config["model_params"]["optimizer"]["defaults"]["lr"] == 0.01
    assert "params" not in config["model_params"]["optimizer"]["param_groups"][0]
    assert config["model_params"]["api_token"] == "<redacted>"
    assert config["random_state"] == 42
    assert config["timing_unit"] == "seconds"
    assert config["environment"]["python"]
    assert "flower" in config["environment"]["libraries"]
    assert set(config["source"]) == {"git_commit", "git_dirty"}

    exported_text = Path(paths["config"]).read_text(encoding="utf-8")
    assert "must-not-be-exported" not in exported_text
    assert "[[0.0" not in exported_text


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


def test_prefixed_losses_are_ordered_beside_matching_accuracies_in_summary_and_csv(tmp_path):
    ordered_metrics = {
        "trained_adv_f1": [(1, 0.4)],
        "attack_loss": [(1, 1.2)],
        "clean_f1": [(1, 0.8)],
        "trained_clean_loss": [(1, 0.7)],
        "attack_accuracy": [(1, 0.5)],
        "trained_adv_accuracy": [(1, 0.3)],
        "clean_loss": [(1, 0.4)],
        "trained_clean_accuracy": [(1, 0.9)],
        "attack_f1": [(1, 0.45)],
        "trained_adv_loss": [(1, 1.8)],
        "clean_accuracy": [(1, 0.95)],
        "trained_clean_f1": [(1, 0.88)],
    }
    result = FALResult(model=PickleableModel(), metrics=ordered_metrics)

    summary_header = result.summary().splitlines()[5].split()
    assert summary_header == [
        "round",
        "clean_accuracy", "clean_loss", "clean_f1",
        "attack_accuracy", "attack_loss", "attack_f1",
        "trained_clean_accuracy", "trained_clean_loss", "trained_clean_f1",
        "trained_adv_accuracy", "trained_adv_loss", "trained_adv_f1",
    ]

    paths = result.save_outputs(str(tmp_path / "results"), save_model=False)
    with open(paths["metrics"], newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == summary_header
        rows = list(reader)

    assert "loss" not in rows[0]
    assert rows[0]["clean_loss"] == "0.4"

    progress_names = _ordered_progress_metrics(ordered_metrics)
    progress_header = _format_header(progress_names, include_loss=False).split()
    assert progress_header == ["round", "clients", *summary_header[1:], "time"]
    assert "loss" not in progress_header
