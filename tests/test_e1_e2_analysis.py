import csv
import json

import pytest

from experiments.analysis import build_e1_e2_report


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_e1_e2_report_preserves_evaluation_protocols(tmp_path):
    campaigns = {}
    for index, training in enumerate(("clean", "attack_only", "adversarial"), start=1):
        campaign = tmp_path / training
        campaigns[training] = campaign
        _write_csv(
            campaign / "runs.csv",
            [
                {
                    "status": "ok",
                    "final_loss": index,
                    "total_execution_time": index * 10,
                    "metric_accuracy": index / 10,
                },
                {
                    "status": "ok",
                    "final_loss": index + 1,
                    "total_execution_time": index * 20,
                    "metric_accuracy": index / 5,
                },
            ],
        )
        _write_csv(
            campaign / "round_metrics.csv",
            [
                {
                    "experiment": training,
                    "seed": 42,
                    "round": 1,
                    "loss": index,
                    "metric_accuracy": index / 10,
                }
            ],
        )

    output = tmp_path / "report"
    paths = build_e1_e2_report(campaigns, output)

    with (output / "e1_final_summary.csv").open(newline="", encoding="utf-8") as file:
        final_rows = list(csv.DictReader(file))
    with (output / "e2_convergence_summary.csv").open(newline="", encoding="utf-8") as file:
        convergence_rows = list(csv.DictReader(file))
    notes = json.loads((output / "comparison_notes.json").read_text(encoding="utf-8"))

    clean_accuracy = next(
        row
        for row in final_rows
        if row["training"] == "clean" and row["measure"] == "metric_accuracy"
    )
    adversarial_accuracy = next(
        row
        for row in convergence_rows
        if row["training"] == "adversarial" and row["measure"] == "metric_accuracy"
    )
    assert float(clean_accuracy["mean"]) == pytest.approx(0.15)
    assert adversarial_accuracy["evaluation"] == "mixed_clean_pgd_adversarial"
    assert len(notes["warnings"]) == 5
    assert paths["e1_final_summary"].endswith("e1_final_summary.csv")
