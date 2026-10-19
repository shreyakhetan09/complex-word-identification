#!/usr/bin/env python3
"""Train and evaluate the complex-word identification models.

Example
-------
    python scripts/train.py \
        --train data/sample/train.tsv \
        --dev   data/sample/dev.tsv \
        --ngram data/ngram_counts.txt.gz \
        --out   model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cwi.baselines import FrequencyThreshold, LengthThreshold, all_complex
from cwi.data import load_ngram_counts, load_word_file
from cwi.metrics import evaluate
from cwi.model import ComplexWordClassifier


def _fmt(name: str, scores: dict) -> str:
    return (
        f"{name:<22} P={scores['precision']:.4f}  "
        f"R={scores['recall']:.4f}  F1={scores['f1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="labeled training file")
    parser.add_argument("--dev", required=True, help="labeled development file")
    parser.add_argument("--ngram", default=None, help="Google n-gram counts file")
    parser.add_argument("--out", default=None, help="where to save the fitted model")
    parser.add_argument("--no-tune", action="store_true", help="skip threshold tuning")
    args = parser.parse_args()

    ngram_counts = load_ngram_counts(args.ngram)
    train = load_word_file(args.train)
    dev = load_word_file(args.dev)
    print(f"loaded {len(train)} train / {len(dev)} dev words\n")

    print("Baselines (dev set)")
    print("-" * 60)
    print(_fmt("majority class", evaluate(dev.labels, all_complex(dev.words))))
    length = LengthThreshold().fit(train.words, train.labels)
    print(_fmt(f"word length >= {length.threshold}", evaluate(dev.labels, length.predict(dev.words))))
    freq = FrequencyThreshold(ngram_counts).fit(train.words, train.labels)
    print(_fmt("word frequency", evaluate(dev.labels, freq.predict(dev.words))))

    print("\nStacked classifier")
    print("-" * 60)
    clf = ComplexWordClassifier(ngram_counts=ngram_counts)
    clf.fit(train)
    if not args.no_tune:
        t = clf.tune_threshold(dev)
        print(f"tuned decision threshold = {t:.3f}")
    print(_fmt("blend (dev)", evaluate(dev.labels, clf.predict(dev))))

    if args.out:
        import joblib

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, args.out)
        print(f"\nsaved model -> {args.out}")


if __name__ == "__main__":
    main()
