"""Feature extraction for complex-word identification.

Each target word is turned into a fixed-length numeric vector built from three
complementary signals:

* **Surface / orthographic** - length, vowel ratio, longest consonant run,
  rare digraphs and letters, inflection flags.
* **Lexical** - log corpus frequency, out-of-vocabulary flag, syllable count,
  and WordNet synset / synonym counts (sense ambiguity).
* **Contextual** - statistics of the sentence the word appears in: sentence
  length, mean word length and frequency, and the target's relative position.

Optional dependencies (``nltk`` for WordNet, ``syllables`` for syllable
estimation) are used when available and fall back to lightweight heuristics
otherwise, so the extractor always runs.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

import numpy as np

VOWELS = set("aeiouy")

COMPLEX_SUFFIXES = (
    "ization", "isation", "ological", "ology", "ography", "otomy", "ectomy",
    "tion", "sion", "ment", "ness", "ity", "ous", "ious", "eous", "ive",
    "able", "ible", "ance", "ence", "ancy", "ency", "ism", "ist", "ize", "ise",
    "ate", "ify", "phy", "graph", "scope", "meter", "metry", "phobia", "pathy",
    "itis", "osis", "oma", "emia", "genic", "lysis",
)
COMPLEX_PREFIXES = (
    "anti", "auto", "inter", "intra", "trans", "super", "hyper", "hypo",
    "pseudo", "quasi", "multi", "micro", "macro", "neuro", "psycho", "socio",
    "bio", "geo", "hydro", "thermo", "photo", "electro", "circum", "contra",
    "extra", "infra", "ultra", "proto", "hetero", "homo", "meta",
)
RARE_DIGRAPHS = ("ph", "gh", "qu", "sch", "chr", "rh", "gn", "kn", "ps", "pt", "ts")

WORD_FEATURE_NAMES = [
    "length", "log_freq", "is_oov", "syllables", "wn_synsets", "wn_lemmas",
    "vowel_ratio", "max_consonant_run", "unique_letter_ratio", "hyphens",
    "has_complex_suffix", "has_complex_prefix", "rare_digraphs", "has_ph_th",
    "has_rare_letter", "plural_s", "inflected", "very_long", "very_short",
    "syllables_per_char",
]
SENTENCE_FEATURE_NAMES = [
    "sent_len", "sent_mean_word_len", "sent_mean_log_freq",
    "target_rel_pos", "sent_min_log_freq", "sent_long_token_ratio",
]
FEATURE_NAMES = WORD_FEATURE_NAMES + SENTENCE_FEATURE_NAMES


def _wordnet_module():
    """Return the WordNet corpus reader, or ``None`` if nltk is unavailable."""
    try:  # pragma: no cover - depends on local install
        from nltk.corpus import wordnet

        wordnet.synsets("test")
        return wordnet
    except Exception:
        return None


def _syllables_module():
    """Return the ``syllables`` module, or ``None`` if unavailable."""
    try:  # pragma: no cover
        import syllables

        return syllables
    except Exception:
        return None


def _max_consonant_run(word: str) -> int:
    best = run = 0
    for ch in word:
        if ch.isalpha() and ch not in VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


class FeatureExtractor:
    """Turns words (and their sentences) into feature matrices."""

    def __init__(self, ngram_counts: Dict[str, int] | None = None) -> None:
        self.ngram_counts = ngram_counts or {}
        self._wn_cache: Dict[str, tuple] = {}

    def __getstate__(self):
        # Keep the object picklable: never persist the lazy module handles,
        # only the counts and a (rebuildable) cache.
        return {"ngram_counts": self.ngram_counts, "_wn_cache": self._wn_cache}

    def __setstate__(self, state):
        self.ngram_counts = state.get("ngram_counts", {})
        self._wn_cache = state.get("_wn_cache", {})

    # -- individual signals -------------------------------------------------
    def _freq(self, word: str) -> int:
        return self.ngram_counts.get(word, 0)

    def _wn_stats(self, word: str) -> tuple:
        wordnet = _wordnet_module()
        if wordnet is None:
            return (0, 0)
        if word not in self._wn_cache:
            syns = wordnet.synsets(word)
            lemmas = {l.name() for syn in syns for l in syn.lemmas()}
            self._wn_cache[word] = (len(syns), len(lemmas))
        return self._wn_cache[word]

    def _syllable_count(self, word: str, vowels: int) -> int:
        syllables = _syllables_module()
        if syllables is not None:
            try:
                return int(syllables.estimate(word))
            except Exception:
                pass
        return max(1, vowels)

    # -- feature rows -------------------------------------------------------
    def word_features(self, word: str) -> List[float]:
        w = word.lower()
        letters = [c for c in w if c.isalpha()]
        n = max(len(letters), 1)
        vowels = sum(c in VOWELS for c in letters)
        freq = self._freq(w)
        n_syn, n_lem = self._wn_stats(w)
        syll = self._syllable_count(w, vowels)
        return [
            float(len(w)),
            math.log1p(freq),
            float(freq == 0),
            float(syll),
            float(n_syn),
            float(n_lem),
            vowels / n,
            float(_max_consonant_run(w)),
            len(set(letters)) / n,
            float(w.count("-")),
            float(any(w.endswith(s) for s in COMPLEX_SUFFIXES)),
            float(any(w.startswith(p) for p in COMPLEX_PREFIXES)),
            float(sum(w.count(d) for d in RARE_DIGRAPHS)),
            float("ph" in w or "th" in w),
            float(any(c in w for c in "xzjq")),
            float(w.endswith("s") and not w.endswith("ss")),
            float(w.endswith("ed") or w.endswith("ing")),
            float(len(w) >= 10),
            float(len(w) <= 4),
            float(syll) / max(len(w), 1),
        ]

    def sentence_features(self, word: str, sentence: str) -> List[float]:
        toks = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", (sentence or "").lower())
        if not toks:
            return [0.0] * len(SENTENCE_FEATURE_NAMES)
        lens = [len(t) for t in toks]
        freqs = [math.log1p(self._freq(t)) for t in toks]
        w = word.lower()
        try:
            pos = toks.index(w)
        except ValueError:
            pos = next((i for i, t in enumerate(toks) if t in w or w in t), len(toks) // 2)
        rel = pos / max(len(toks) - 1, 1)
        return [
            float(len(toks)),
            float(np.mean(lens)),
            float(np.mean(freqs)),
            float(rel),
            float(np.min(freqs)),
            float(sum(l >= 8 for l in lens) / len(toks)),
        ]

    def transform(
        self, words: Sequence[str], sentences: Sequence[str] | None = None
    ) -> np.ndarray:
        """Build an ``(n_words, n_features)`` matrix."""
        if sentences is None:
            sentences = [""] * len(words)
        rows = [
            self.word_features(w) + self.sentence_features(w, s)
            for w, s in zip(words, sentences)
        ]
        return np.asarray(rows, dtype=float)
