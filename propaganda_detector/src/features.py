"""
NLP feature extraction for propaganda / media-bias detection.

Feature groups
--------------
1. TF-IDF unigrams + bigrams (sparse, high-dimensional)
2. Linguistic surface features (dense, 20-ish dimensions):
   - Sentiment (VADER compound, pos, neg, neu)
   - Readability scores (Flesch, Gunning Fog, etc.)
   - Lexical diversity
   - Loaded / emotional language word counts
   - Pronoun usage (1st/2nd person)
   - Punctuation patterns (!!!, ???, ALL-CAPS ratio)
   - Hedging vs. assertive language ratios
   - Passive voice indicator
   - Named-entity density
   - Average word/sentence length

The two feature sets are combined into a single feature matrix.
"""

from __future__ import annotations

import re
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# NLTK resources (downloaded lazily)
import nltk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy NLTK downloads
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)

_ensure_nltk_resources()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    logger.warning("textstat not installed — readability features disabled.")

_vader = SentimentIntensityAnalyzer()
_stopwords = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Propaganda-related word lists
# ---------------------------------------------------------------------------

LOADED_WORDS = {
    "terrorist", "extremist", "radical", "corrupt", "evil", "criminal",
    "lie", "liar", "fraud", "hoax", "conspiracy", "fake", "regime",
    "invasion", "crisis", "disaster", "catastrophe", "threat", "danger",
    "illegal", "disgrace", "shame", "outrage", "shocking", "alarming",
    "unprecedented", "devastating", "terrible", "horrible", "awful",
    "wonderful", "amazing", "incredible", "best", "worst", "greatest",
    "unbelievable", "disgusting", "pathetic", "despicable", "deplorable",
    "hero", "savior", "enemy", "traitor", "puppet", "propaganda",
}

HEDGING_WORDS = {
    "perhaps", "maybe", "possibly", "might", "could", "seem", "appear",
    "suggest", "indicate", "likely", "unlikely", "probably", "allegedly",
    "reportedly", "purportedly", "supposedly", "claimed", "according",
    "appears", "seems", "suggests", "indicates",
}

ASSERTIVE_WORDS = {
    "clearly", "obviously", "certainly", "definitely", "undoubtedly",
    "absolutely", "always", "never", "every", "all", "none", "must",
    "will", "proven", "fact", "truth", "lie", "false", "true", "real",
    "fake", "totally", "completely", "entirely", "undeniable",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
PASSIVE_AUX = {"was", "were", "is", "are", "be", "been", "being", "am"}

# ---------------------------------------------------------------------------
# Surface-feature extractor (dense)
# ---------------------------------------------------------------------------

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Produces a dense numpy array of hand-crafted linguistic features.
    Compatible with sklearn Pipeline / FeatureUnion.
    """

    FEATURE_NAMES: List[str] = [
        "vader_compound",
        "vader_pos",
        "vader_neg",
        "vader_neu",
        "flesch_reading_ease",
        "gunning_fog",
        "avg_word_len",
        "avg_sent_len_words",
        "num_sentences",
        "num_words",
        "type_token_ratio",
        "loaded_word_ratio",
        "hedging_ratio",
        "assertive_ratio",
        "first_person_ratio",
        "second_person_ratio",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "passive_voice_ratio",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([self._extract(text) for text in X])

    def _extract(self, text: str) -> np.ndarray:
        text = str(text)
        tokens_raw = word_tokenize(text)
        tokens = [t.lower() for t in tokens_raw if t.isalpha()]
        sents = sent_tokenize(text)
        n_words = max(len(tokens), 1)
        n_sents = max(len(sents), 1)

        # --- Sentiment ---
        vs = _vader.polarity_scores(text)

        # --- Readability ---
        if HAS_TEXTSTAT:
            flesch = textstat.flesch_reading_ease(text)
            fog = textstat.gunning_fog(text)
        else:
            flesch, fog = 0.0, 0.0

        # --- Lexical ---
        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
        avg_sent_len = n_words / n_sents
        ttr = len(set(tokens)) / n_words  # type-token ratio

        # --- Propaganda-signal word ratios ---
        loaded_ratio = sum(1 for t in tokens if t in LOADED_WORDS) / n_words
        hedging_ratio = sum(1 for t in tokens if t in HEDGING_WORDS) / n_words
        assertive_ratio = sum(1 for t in tokens if t in ASSERTIVE_WORDS) / n_words
        first_p = sum(1 for t in tokens if t in FIRST_PERSON) / n_words
        second_p = sum(1 for t in tokens if t in SECOND_PERSON) / n_words

        # --- Punctuation ---
        excl = text.count("!")
        quest = text.count("?")
        alpha_chars = max(sum(1 for c in text if c.isalpha()), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / alpha_chars

        # --- Passive voice (heuristic: BE-verb + past participle) ---
        pos_tags = nltk.pos_tag(tokens_raw)
        passive_count = 0
        for i, (word, tag) in enumerate(pos_tags):
            if word.lower() in PASSIVE_AUX and i + 1 < len(pos_tags):
                if pos_tags[i + 1][1] in ("VBN",):
                    passive_count += 1
        passive_ratio = passive_count / n_sents

        return np.array([
            vs["compound"],
            vs["pos"],
            vs["neg"],
            vs["neu"],
            flesch,
            fog,
            avg_word_len,
            avg_sent_len,
            n_sents,
            n_words,
            ttr,
            loaded_ratio,
            hedging_ratio,
            assertive_ratio,
            first_p,
            second_p,
            float(excl),
            float(quest),
            caps_ratio,
            passive_ratio,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# TF-IDF transformer (sparse)
# ---------------------------------------------------------------------------

class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Wraps TfidfVectorizer for use inside a FeatureUnion."""

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range=(1, 2),
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
            stop_words="english",
        )

    def fit(self, X, y=None):
        self._vec.fit(X)
        return self

    def transform(self, X):
        return self._vec.transform(X)

    def get_feature_names_out(self):
        return self._vec.get_feature_names_out()


# ---------------------------------------------------------------------------
# Dense-wrapper so FeatureUnion can combine sparse + dense
# ---------------------------------------------------------------------------

class DenseTransformer(BaseEstimator, TransformerMixin):
    """Converts output of LinguisticFeatureExtractor to float64 ndarray."""

    def __init__(self, extractor: LinguisticFeatureExtractor):
        self.extractor = extractor

    def fit(self, X, y=None):
        self.extractor.fit(X, y)
        return self

    def transform(self, X):
        arr = self.extractor.transform(X)
        return arr.astype(np.float64)


# ---------------------------------------------------------------------------
# Combined feature pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(max_tfidf_features: int = 15_000) -> FeatureUnion:
    """
    Returns a FeatureUnion that combines:
      - TF-IDF unigrams/bigrams  (sparse → converted to dense via hstack)
      - Linguistic surface features (dense)

    Note: sklearn FeatureUnion handles sparse+dense via hstack internally
    (the sparse matrix is kept sparse; dense is added).
    """
    ling_extractor = LinguisticFeatureExtractor()

    union = FeatureUnion(
        transformer_list=[
            ("tfidf", TfidfTransformer(max_features=max_tfidf_features)),
            ("linguistic", ling_extractor),
        ]
    )
    return union


def get_feature_names(pipeline: FeatureUnion) -> List[str]:
    tfidf_names = list(pipeline.transformer_list[0][1].get_feature_names_out())
    ling_names = LinguisticFeatureExtractor.FEATURE_NAMES
    return tfidf_names + ling_names
