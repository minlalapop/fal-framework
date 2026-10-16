from experiments.benchmark.e5_v2_instrumentation import FlowerTimingCollector
from experiments.benchmark.e5_v2_runner import (
    build_balanced_schedule,
    compare_paired_systems,
    verify_paired_equivalence,
)


def test_v2_phase_collector_partitions_round(monkeypatch):
    timestamps = iter(float(value) for value in range(8))
    monkeypatch.setattr(
        "experiments.benchmark.e5_v2_instrumentation.time.perf_counter",
        lambda: next(timestamps),
    )
    collector = FlowerTimingCollector()
    for name in (
        "configure_fit_start",
        "configure_fit_end",
        "aggregate_fit_start",
        "aggregate_fit_end",
        "configure_evaluate_start",
        "configure_evaluate_end",
        "aggregate_evaluate_start",
        "aggregate_evaluate_end",
    ):
        collector.mark(1, name)

    phase = collector.round_phases()[0]

    assert phase["round_execution_time"] == 7.0
    assert phase["fit_window_time"] == 1.0
    assert phase["evaluate_window_time"] == 1.0
    assert phase["aggregate_fit_time"] == 1.0
    assert phase["aggregate_evaluate_time"] == 1.0


def test_v2_schedule_is_paired_balanced_and_reproducible():
    first = build_balanced_schedule(
        seeds=(42, 123, 2026),
        modes=("clean", "adversarial"),
        repetitions=3,
        schedule_seed=17,
    )
    second = build_balanced_schedule(
        seeds=(42, 123, 2026),
        modes=("clean", "adversarial"),
        repetitions=3,
        schedule_seed=17,
    )

    assert first == second
    assert len(first) == 18
    assert all(set(block["system_order"]) == {"fal", "flower_art"} for block in first)
    assert sum(block["system_order"][0] == "fal" for block in first) == 9


def test_v2_equivalence_and_paired_comparison():
    common = {
        "mode": "clean",
        "seed": 42,
        "repetition": 1,
        "block_id": "clean_seed_42_rep_1",
        "initial_parameter_digest": "weights",
        "partition_digest": "partition",
        "final_loss": 0.1,
        "final_accuracy": 0.9,
        "final_f1": 0.9,
        "end_to_end_wall_time": 2.0,
    }
    rows = [
        {**common, "system": "fal"},
        {
            **common,
            "system": "flower_art",
            "end_to_end_wall_time": 1.0,
        },
    ]

    equivalence = verify_paired_equivalence(rows)
    comparison = compare_paired_systems(rows)

    assert equivalence[0]["equivalent"] is True
    wall = next(row for row in comparison if row["measure"] == "end_to_end_wall_time")
    assert wall["paired_difference_mean"] == 1.0
    assert wall["difference_percent"] == 100.0
