import csv
from pathlib import Path

from experiments.benchmark.e5_report import ARTICLE_MEASURES, build_e5_report


def test_build_e5_report_creates_article_outputs(tmp_path: Path):
    summary_path = tmp_path / "e5_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("system", "mode", "measure", "n", "mean", "std"),
        )
        writer.writeheader()
        for system, offset in (("fal", 2.0), ("flower_art", 1.0)):
            for mode, factor in (("clean", 1.0), ("adversarial", 1.5)):
                for measure, _, _ in ARTICLE_MEASURES:
                    writer.writerow(
                        {
                            "system": system,
                            "mode": mode,
                            "measure": measure,
                            "n": 3,
                            "mean": offset * factor,
                            "std": 0.1,
                        }
                    )
    with (tmp_path / "e5_adversarial_overhead.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "system",
                "measure",
                "clean_mean",
                "adversarial_mean",
                "overhead_percent",
            ),
        )
        writer.writeheader()
        for system in ("fal", "flower_art"):
            for measure in ("total_wall_time", "cpu_time", "peak_ram_mb"):
                writer.writerow(
                    {
                        "system": system,
                        "measure": measure,
                        "clean_mean": 1.0,
                        "adversarial_mean": 1.5,
                        "overhead_percent": 50.0,
                    }
                )

    outputs = build_e5_report(tmp_path)

    assert Path(outputs["table_csv"]).exists()
    assert "mean ± sample standard deviation" in Path(outputs["table_markdown"]).read_text()
    assert "<svg" in Path(outputs["figure"]).read_text()
