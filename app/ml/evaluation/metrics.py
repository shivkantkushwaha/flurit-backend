"""Evaluation metrics calculation for binary churn prediction models."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics suite for binary classification."""

    sample_count: int
    positive_count: int
    negative_count: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    brier_score: float
    brier_score_baseline: float
    brier_skill_score: float
    decision_threshold: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold_tuned: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return asdict(self)


def evaluate_binary_classifier(
    y_true: npt.NDArray[np.int_] | list[int],
    y_prob: npt.NDArray[np.float64] | list[float],
    threshold: float = 0.5,
) -> ModelMetrics:
    """Compute comprehensive performance metrics for binary churn classifier.

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted positive-class probabilities P(Y=1).
        threshold: Decision cutoff for hard binary predictions (default: 0.5).

    Returns:
        ModelMetrics dataclass containing all computed indicators.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    n_samples = len(y_t)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    prevalence = float(n_pos / n_samples) if n_samples > 0 else 0.0

    # Probability-based metrics (ranking and calibration)
    pr_auc = float(average_precision_score(y_t, y_p)) if n_pos > 0 else 0.0
    roc_auc = float(roc_auc_score(y_t, y_p)) if (n_pos > 0 and n_neg > 0) else 0.5
    brier = float(brier_score_loss(y_t, y_p))
    brier_baseline = float(prevalence * (1.0 - prevalence))
    brier_skill_score = float(1.0 - (brier / brier_baseline)) if brier_baseline > 0.0 else 0.0

    # Threshold-based hard classification metrics
    y_pred = (y_p >= threshold).astype(int)
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return ModelMetrics(
        sample_count=n_samples,
        positive_count=n_pos,
        negative_count=n_neg,
        prevalence=prevalence,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=brier,
        brier_score_baseline=brier_baseline,
        brier_skill_score=brier_skill_score,
        decision_threshold=threshold,
        precision=prec,
        recall=rec,
        f1=f1,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        threshold_tuned=False,
    )
