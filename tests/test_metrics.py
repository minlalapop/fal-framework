from fal.evaluation.metrics import MetricFactory


def test_metric_factory_computes_scalar_metrics():
    manager = MetricFactory.create([
        "accuracy",
        {"name": "f1", "average": "macro"},
    ])

    values = manager.compute_all(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
    )

    assert values["accuracy"] == 0.75
    assert 0 <= values["f1"] <= 1


def test_metric_factory_computes_detailed_metrics():
    manager = MetricFactory.create([
        "confusion_matrix",
        "classification_report",
    ])

    values = manager.compute_all(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
    )

    assert values["confusion_matrix"] == [[2, 0], [1, 1]]
    assert isinstance(values["classification_report"], dict)
    assert "accuracy" in values["classification_report"]
