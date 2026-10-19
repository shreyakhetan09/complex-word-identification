"""Binary classification metrics for complex-word identification.

The positive class (label ``1``) means "complex". All three functions take the
ground-truth labels first and the predicted labels second.
"""

from typing import Sequence


def get_precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of predicted-complex words that are actually complex."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def get_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of truly complex words that we recovered."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_fscore(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Harmonic mean of precision and recall (F1)."""
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Return precision, recall and F1 as a dictionary."""
    return {
        "precision": get_precision(y_true, y_pred),
        "recall": get_recall(y_true, y_pred),
        "f1": get_fscore(y_true, y_pred),
    }
