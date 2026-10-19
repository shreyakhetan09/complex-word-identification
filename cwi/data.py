"""Dataset loading utilities for complex-word identification.

Two file formats are supported:

* **Labeled / unlabeled word files** - tab separated, one target word per line
  with an optional header. The first column is the word and the second column
  (when present) is the binary label. A fourth column, if present, holds the
  sentence the word was drawn from and is used for the contextual features::

      word    label   ...   sentence
      cat     0             The cat sat on the mat .
      abdicate 1            The king chose to abdicate the throne .

* **Google n-gram counts** - a (optionally gzip-compressed) two column file of
  ``token<TAB>count`` used by the frequency features.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class WordDataset:
    """A collection of target words with optional labels and sentences."""

    words: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def has_labels(self) -> bool:
        return len(self.labels) == len(self.words) and len(self.words) > 0


def _open(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf8")
    return open(path, "rt", encoding="utf8")


def load_word_file(
    path: str | Path, labeled: bool = True, has_header: bool = True
) -> WordDataset:
    """Load a labeled or unlabeled word file into a :class:`WordDataset`."""
    words: List[str] = []
    labels: List[int] = []
    sentences: List[str] = []

    with _open(path) as handle:
        for i, raw in enumerate(handle):
            if has_header and i == 0:
                continue
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            words.append(parts[0].lower())
            if labeled and len(parts) > 1:
                labels.append(int(parts[1]))
            sentences.append(parts[3] if len(parts) > 3 else "")

    return WordDataset(words=words, labels=labels, sentences=sentences)


def load_ngram_counts(path: Optional[str | Path]) -> Dict[str, int]:
    """Load Google n-gram unigram counts into a ``{token: count}`` dict.

    Returns an empty mapping when ``path`` is ``None`` so the frequency features
    degrade gracefully when the (large) counts file is unavailable.
    """
    counts: Dict[str, int] = defaultdict(int)
    if path is None:
        return counts
    with _open(path) as handle:
        for line in handle:
            token, _, count = line.strip().partition("\t")
            if token and count and token[0].islower():
                counts[token] = int(count)
    return counts
