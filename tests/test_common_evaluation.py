import pytest

from experiments.analysis.common_evaluation import summarize_common_evaluation


def test_common_evaluation_summary_groups_training_protocols():
    rows = [
        {
            "training": "clean",
            "seed": 1,
            "model_path": "one.pkl",
            "test_examples": 10,
            "clean_accuracy": 0.8,
            "pgd_accuracy": 0.4,
        },
        {
            "training": "clean",
            "seed": 2,
            "model_path": "two.pkl",
            "test_examples": 10,
            "clean_accuracy": 1.0,
            "pgd_accuracy": 0.6,
        },
        {
            "training": "adversarial",
            "seed": 1,
            "model_path": "three.pkl",
            "test_examples": 10,
            "clean_accuracy": 0.7,
            "pgd_accuracy": 0.65,
        },
    ]

    summary = summarize_common_evaluation(rows)
    clean_accuracy = next(
        row
        for row in summary
        if row["training"] == "clean" and row["measure"] == "clean_accuracy"
    )
    robust_accuracy = next(
        row
        for row in summary
        if row["training"] == "clean" and row["measure"] == "pgd_accuracy"
    )

    assert clean_accuracy["mean"] == pytest.approx(0.9)
    assert clean_accuracy["std"] == pytest.approx(0.141421356)
    assert robust_accuracy["mean"] == pytest.approx(0.5)
