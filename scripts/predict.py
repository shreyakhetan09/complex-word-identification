#!/usr/bin/env python3
"""Predict simple/complex labels for a list of words.

Reads a word file (optionally with sentence context) and writes one label per
line: ``1`` for complex, ``0`` for simple.

Example
-------
    python scripts/predict.py --model model.joblib \
        --input data/sample/test.tsv --output predictions.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from cwi.data import load_word_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model saved by train.py")
    parser.add_argument("--input", required=True, help="word file to label")
    parser.add_argument("--output", default=None, help="file to write labels to")
    parser.add_argument(
        "--unlabeled", action="store_true", help="input has no label column"
    )
    args = parser.parse_args()

    clf = joblib.load(args.model)
    data = load_word_file(args.input, labeled=not args.unlabeled)
    preds = clf.predict(data)

    if args.output:
        with open(args.output, "w", encoding="utf8") as handle:
            handle.write("\n".join(map(str, preds)) + "\n")
        complex_rate = sum(preds) / len(preds) if preds else 0.0
        print(f"wrote {len(preds)} labels -> {args.output} (complex rate {complex_rate:.2%})")
    else:
        for word, label in zip(data.words, preds):
            tag = "complex" if label else "simple"
            print(f"{word}\t{tag}")


if __name__ == "__main__":
    main()
