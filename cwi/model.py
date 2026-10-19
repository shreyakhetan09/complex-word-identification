"""The learned complex-word classifier.

A soft-voting blend of gradient-boosted trees, a random forest, and logistic
regression over the features in :mod:`cwi.features`. Trees capture the
non-linear interactions between morphology and frequency; the linear model
keeps the decision boundary stable on the sparse tail. Probabilities are
averaged and thresholded, with the threshold tuned on a validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import WordDataset
from .features import FeatureExtractor


@dataclass
class BlendWeights:
    gradient_boosting: float = 0.45
    random_forest: float = 0.35
    logistic: float = 0.20


class ComplexWordClassifier:
    """Stacked classifier for complex-word identification."""

    def __init__(
        self,
        ngram_counts: Dict[str, int] | None = None,
        weights: BlendWeights | None = None,
        threshold: float = 0.51,
        random_state: int = 42,
    ) -> None:
        self.extractor = FeatureExtractor(ngram_counts)
        self.weights = weights or BlendWeights()
        self.threshold = threshold
        self.random_state = random_state
        self.scaler = StandardScaler()

        self.gb = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=250,
            l2_regularization=0.1, random_state=random_state,
        )
        self.rf = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        self._fitted = False

    # -- helpers ------------------------------------------------------------
    def _features(self, data: WordDataset) -> np.ndarray:
        return self.extractor.transform(data.words, data.sentences or None)

    def _blend(self, X: np.ndarray) -> np.ndarray:
        w = self.weights
        return (
            w.gradient_boosting * self.gb.predict_proba(X)[:, 1]
            + w.random_forest * self.rf.predict_proba(X)[:, 1]
            + w.logistic * self.lr.predict_proba(X)[:, 1]
        )

    # -- public API ---------------------------------------------------------
    def fit(self, data: WordDataset) -> "ComplexWordClassifier":
        if not data.has_labels:
            raise ValueError("training data must have labels")
        X = self.scaler.fit_transform(self._features(data))
        y = np.asarray(data.labels)
        self.gb.fit(X, y)
        self.rf.fit(X, y)
        self.lr.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, data: WordDataset) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before predicting")
        X = self.scaler.transform(self._features(data))
        return self._blend(X)

    def predict(self, data: WordDataset) -> List[int]:
        proba = self.predict_proba(data)
        return (proba >= self.threshold).astype(int).tolist()

    def tune_threshold(
        self, data: WordDataset, grid: Sequence[float] | None = None
    ) -> float:
        """Pick the decision threshold that maximises F1 on ``data``."""
        from .metrics import get_fscore

        if not data.has_labels:
            raise ValueError("threshold tuning needs labeled data")
        grid = grid if grid is not None else np.linspace(0.2, 0.8, 61)
        proba = self.predict_proba(data)
        best_t, best_f = self.threshold, -1.0
        for t in grid:
            preds = (proba >= t).astype(int).tolist()
            f = get_fscore(data.labels, preds)
            if f > best_f:
                best_f, best_t = f, float(t)
        self.threshold = best_t
        return best_t
