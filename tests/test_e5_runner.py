import pytest

from experiments.benchmark.e5_runner import (
    adversarial_overheads,
    compare_systems,
    summarize_e5_runs,
)


def _rows():
    rows = []
    for system, offset in (("fal", 2.0), ("flower_art", 1.0)):
        for mode, factor in (("clean", 1.0), ("adversarial", 2.0)):
            for seed in (1, 2):
                rows.append(
                    {
                        "system": system,
                        "mode": mode,
                        "total_wall_time": offset * factor * seed,
                        "cpu_time": offset * factor * seed,
                        "peak_ram_mb": offset * factor * seed,
                    }
                )
    return rows


def test_e5_summary_comparison_and_overhead():
    summary = summarize_e5_runs(_rows())
    comparisons = compare_systems(summary)
    overheads = adversarial_overheads(summary)

    clean_wall = next(
        row
        for row in comparisons
        if row["mode"] == "clean" and row["measure"] == "total_wall_time"
    )
    fal_overhead = next(
        row
        for row in overheads
        if row["system"] == "fal" and row["measure"] == "total_wall_time"
    )
    assert clean_wall["difference_percent"] == pytest.approx(100.0)
    assert fal_overhead["overhead_percent"] == pytest.approx(100.0)
