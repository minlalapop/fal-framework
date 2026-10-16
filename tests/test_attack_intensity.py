from pathlib import Path

import pytest

from experiments.analysis.attack_intensity import (
    _validate_epsilons,
    _write_accuracy_svg,
    summarize_attack_intensity,
)


def test_attack_intensity_summary_and_svg(tmp_path):
    rows = [
        {"training": "clean", "seed": 1, "epsilon": 0.1, "accuracy": 0.8},
        {"training": "clean", "seed": 2, "epsilon": 0.1, "accuracy": 0.6},
        {"training": "adversarial", "seed": 1, "epsilon": 0.1, "accuracy": 0.7},
    ]

    summary = summarize_attack_intensity(rows)
    clean = next(row for row in summary if row["training"] == "clean")
    assert clean["mean"] == pytest.approx(0.7)
    assert clean["std"] == pytest.approx(0.141421356)

    figure = tmp_path / "figure.svg"
    _write_accuracy_svg(figure, summary)
    assert figure.read_text(encoding="utf-8").startswith("<svg")


def test_attack_epsilons_are_positive_and_deduplicated():
    assert _validate_epsilons([0.2, 0.1, 0.2]) == [0.1, 0.2]
    with pytest.raises(ValueError, match="positive"):
        _validate_epsilons([0.0])
