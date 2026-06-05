"""Concrete metric wrappers around ``sklearn.metrics``."""

from __future__ import annotations

from typing import Any

from sklearn import metrics as sklearn_metrics

from fal.evaluation.metrics.base import BaseMetric


class AccuracyMetric(BaseMetric):
    name = "accuracy"

    def __init__(self) -> None:
        self.metrics = sklearn_metrics

    def compute(self, y_true: Any, y_pred: Any) -> float:
        return float(self.metrics.accuracy_score(y_true, y_pred))


class F1Metric(BaseMetric):
    name = "f1"

    def __init__(self, average: str = "weighted", zero_division: int = 0) -> None:
        self.metrics = sklearn_metrics
        self.average = average
        self.zero_division = zero_division

    def compute(self, y_true: Any, y_pred: Any) -> float:
        return float(
            self.metrics.f1_score(
                y_true,
                y_pred,
                average=self.average,
                zero_division=self.zero_division,
            )
        )


class PrecisionMetric(BaseMetric):
    name = "precision"

    def __init__(self, average: str = "weighted", zero_division: int = 0) -> None:
        self.metrics = sklearn_metrics
        self.average = average
        self.zero_division = zero_division

    def compute(self, y_true: Any, y_pred: Any) -> float:
        return float(
            self.metrics.precision_score(
                y_true,
                y_pred,
                average=self.average,
                zero_division=self.zero_division,
            )
        )


class RecallMetric(BaseMetric):
    name = "recall"

    def __init__(self, average: str = "weighted", zero_division: int = 0) -> None:
        self.metrics = sklearn_metrics
        self.average = average
        self.zero_division = zero_division

    def compute(self, y_true: Any, y_pred: Any) -> float:
        return float(
            self.metrics.recall_score(
                y_true,
                y_pred,
                average=self.average,
                zero_division=self.zero_division,
            )
        )


class AUCMetric(BaseMetric):
    name = "auc"

    def __init__(self, **kwargs: Any) -> None:
        self.metrics = sklearn_metrics
        self.kwargs = kwargs

    def compute(self, y_true: Any, y_pred: Any) -> float:
        return float(self.metrics.roc_auc_score(y_true, y_pred, **self.kwargs))


class ConfusionMatrixMetric(BaseMetric):
    name = "confusion_matrix"

    def __init__(self, labels: list[Any] | None = None) -> None:
        self.metrics = sklearn_metrics
        self.labels = labels

    def compute(self, y_true: Any, y_pred: Any) -> list[list[int]]:
        matrix = self.metrics.confusion_matrix(
            y_true,
            y_pred,
            labels=self.labels,
        )
        return matrix.tolist()


class ClassificationReportMetric(BaseMetric):
    name = "classification_report"

    def __init__(
        self,
        labels: list[Any] | None = None,
        target_names: list[str] | None = None,
        digits: int = 2,
        output_dict: bool = True,
        zero_division: int = 0,
    ) -> None:
        self.metrics = sklearn_metrics
        self.labels = labels
        self.target_names = target_names
        self.digits = digits
        self.output_dict = output_dict
        self.zero_division = zero_division

    def compute(self, y_true: Any, y_pred: Any) -> dict[str, Any] | str:
        return self.metrics.classification_report(
            y_true,
            y_pred,
            labels=self.labels,
            target_names=self.target_names,
            digits=self.digits,
            output_dict=self.output_dict,
            zero_division=self.zero_division,
        )
