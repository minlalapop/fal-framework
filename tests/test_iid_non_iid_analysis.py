import pytest

from experiments.analysis.iid_non_iid import (
    _write_convergence_svg,
    _write_heatmap_svg,
    summarize_e4_convergence,
)


def test_e4_convergence_summary_and_heatmap(tmp_path):
    rounds = [
        {"distribution": "iid", "seed": 1, "round": 1, "metric_accuracy": 0.8},
        {"distribution": "iid", "seed": 2, "round": 1, "metric_accuracy": 1.0},
        {"distribution": "non_iid", "seed": 1, "round": 1, "metric_accuracy": 0.5},
    ]
    summary = summarize_e4_convergence(rounds)
    iid = next(row for row in summary if row["distribution"] == "iid")
    assert iid["mean"] == pytest.approx(0.9)

    labels = [
        {
            "distribution": distribution,
            "seed": 42,
            "client_id": client_id,
            "label": label,
            "count": 10,
            "client_examples": 100,
            "proportion": 0.1,
        }
        for distribution in ("iid", "non_iid")
        for client_id in range(3)
        for label in range(10)
    ]
    figure = tmp_path / "heatmap.svg"
    _write_heatmap_svg(figure, labels, seed=42)
    content = figure.read_text(encoding="utf-8")
    assert content.startswith("<svg")
    assert "Non-IID" in content

    convergence = tmp_path / "convergence.svg"
    convergence_rows = []
    for distribution in ("iid", "non_iid"):
        for round_number in range(1, 6):
            for measure, mean in (("metric_accuracy", 0.8), ("loss", 0.5)):
                convergence_rows.append(
                    {
                        "distribution": distribution,
                        "round": round_number,
                        "measure": measure,
                        "n": 3,
                        "mean": mean,
                        "std": 0.05,
                    }
                )
    _write_convergence_svg(convergence, convergence_rows)
    assert "federated convergence" in convergence.read_text(encoding="utf-8")
