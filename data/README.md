# Data

A small, runnable sample lives in [`sample/`](sample/) so the project works out
of the box. Each file is tab separated.

### Labeled files (`train.tsv`, `dev.tsv`)

| column | meaning |
| ------ | ------- |
| `word` | the target word |
| `label` | `1` = complex, `0` = simple |
| `n_annotators` | metadata (ignored by the model) |
| `sentence` | the sentence the word was drawn from, used for context features |

### Unlabeled file (`test.tsv`)

Same as above without the `label` column. Run it through `scripts/predict.py`
with `--unlabeled`.

### Corpus frequencies (optional)

The frequency baseline and the `log_freq` feature read Google Web 1T unigram
counts, a two-column `token<TAB>count` file (optionally gzip-compressed). Point
`--ngram` at it. When omitted, frequency-based signals default to zero and the
rest of the pipeline still runs.

To train on the full Complex Word Identification corpus, drop the SemEval-2016
CWI files (or your own) into this folder using the same layout.
