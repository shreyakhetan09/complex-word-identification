"""Interpretable baselines for complex-word identification.

These exist to put the learned model in context: if a one-line rule already
scores well, the task is easy; if it does not, the features in :mod:`cwi.model`
are doing real work. Each baseline follows a tiny fit/predict interface.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .metrics import get_fscore


def all_complex(words: Sequence[str]) -> List[int]:
    """Majority-class baseline: label every word complex."""
    return [1] * len(words)


class LengthThreshold:
    """Predict "complex" when a word is at least ``threshold`` characters long.

    The threshold is chosen on the training set to maximise F1.
    """

    def __init__(self, min_len: int = 1, max_len: int = 20) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.threshold: Optional[int] = None

    @staticmethod
    def _predict(words: Sequence[str], threshold: int) -> List[int]:
        return [1 if len(w) >= threshold else 0 for w in words]

    def fit(self, words: Sequence[str], labels: Sequence[int]) -> "LengthThreshold":
        best_t, best_f = self.min_len, -1.0
        for t in range(self.min_len, self.max_len + 1):
            f = get_fscore(labels, self._predict(words, t))
            if f > best_f:
                best_f, best_t = f, t
        self.threshold = best_t
        return self

    def predict(self, words: Sequence[str]) -> List[int]:
        if self.threshold is None:
            raise RuntimeError("call fit() before predict()")
        return self._predict(words, self.threshold)


class FrequencyThreshold:
    """Predict "complex" when a word's corpus frequency is below a threshold.

    Rare words tend to be complex. Candidate thresholds are drawn from quantiles
    and a log-spaced grid of the training-set frequencies.
    """

    def __init__(self, ngram_counts: Dict[str, int]) -> None:
        self.ngram_counts = ngram_counts
        self.threshold: Optional[int] = None

    def _freq(self, word: str) -> int:
        return self.ngram_counts.get(word, 0)

    def _predict(self, words: Sequence[str], threshold: int) -> List[int]:
        return [1 if self._freq(w) < threshold else 0 for w in words]

    def fit(self, words: Sequence[str], labels: Sequence[int]) -> "FrequencyThreshold":
        import numpy as np

        freqs = np.array([self._freq(w) for w in words], dtype=float)
        max_f = max(float(freqs.max(initial=0.0)), 2.0)
        candidates = np.unique(
            np.concatenate(
                [
                    np.quantile(freqs, np.linspace(0, 1, 100)).astype(np.int64),
                    np.logspace(0, np.log10(max_f), 80).astype(np.int64),
                    np.array([1], dtype=np.int64),
                ]
            )
        )
        best_t, best_f = int(candidates[0]), -1.0
        for t in candidates:
            f = get_fscore(labels, self._predict(words, int(t)))
            if f > best_f:
                best_f, best_t = f, int(t)
        self.threshold = best_t
        return self

    def predict(self, words: Sequence[str]) -> List[int]:
        if self.threshold is None:
            raise RuntimeError("call fit() before predict()")
        return self._predict(words, self.threshold)
