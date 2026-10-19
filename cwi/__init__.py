"""Complex Word Identification (CWI).

A small, self-contained toolkit for deciding whether a word is "simple" or
"complex" in context - the first step of a lexical text-simplification
pipeline. The package exposes evaluation metrics, interpretable baselines, a
rich feature extractor, and a stacked classifier.
"""

from .metrics import get_precision, get_recall, get_fscore
from .baselines import all_complex, LengthThreshold, FrequencyThreshold
from .model import ComplexWordClassifier

__all__ = [
    "get_precision",
    "get_recall",
    "get_fscore",
    "all_complex",
    "LengthThreshold",
    "FrequencyThreshold",
    "ComplexWordClassifier",
]

__version__ = "1.0.0"
