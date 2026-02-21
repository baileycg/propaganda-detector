"""
Prediction pipeline for new, unseen text.

Given raw text (a sentence, paragraph, article, or URL-scraped body), this
module returns:
  - A binary label  : "Propaganda / Biased" | "Non-biased"
  - A confidence %  : probability from the trained classifier
  - A feature breakdown : which linguistic signals drove the prediction
  - A placeholder summary stub  : reserved for LLM integration

Usage
-----
    from src.predictor import PropagandaDetector
    detector = PropagandaDetector()
    result = detector.predict("The corrupt politicians destroyed our nation!")
    print(result)
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import load_pipeline
from .features import LinguisticFeatureExtractor, LOADED_WORDS, HEDGING_WORDS, ASSERTIVE_WORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    text: str
    label: str                          # "Propaganda / Biased" or "Non-biased"
    confidence: float                   # 0.0 – 1.0
    probability_biased: float           # raw P(biased)
    probability_nonbiased: float        # raw P(non-biased)
    signal_summary: Dict[str, float]    # key linguistic signals
    triggered_words: List[str]          # loaded/assertive/emotional words found
    llm_summary: Optional[str] = None  # placeholder for LLM explanation

    def __str__(self) -> str:
        bar_len = 30
        filled = int(self.confidence * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            "",
            "┌─────────────────────────────────────────────┐",
            f"│  Label      : {self.label:<30}│",
            f"│  Confidence : [{bar}] {self.confidence*100:.1f}%  │",
            f"│  P(biased)  : {self.probability_biased:.4f}                           │",
            "├─────────────────────────────────────────────┤",
            "│  Top linguistic signals:                    │",
        ]
        for sig, val in self.signal_summary.items():
            lines.append(f"│    {sig:<22} : {val:>6.4f}              │")
        if self.triggered_words:
            words_str = ", ".join(self.triggered_words[:8])
            for chunk in textwrap.wrap(f"Triggered words: {words_str}", width=43):
                lines.append(f"│  {chunk:<43}│")
        if self.llm_summary:
            lines.append("├─────────────────────────────────────────────┤")
            lines.append("│  LLM Summary:                               │")
            for chunk in textwrap.wrap(self.llm_summary, width=43):
                lines.append(f"│  {chunk:<43}│")
        lines.append("└─────────────────────────────────────────────┘")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probability_biased": round(self.probability_biased, 4),
            "probability_nonbiased": round(self.probability_nonbiased, 4),
            "signal_summary": {k: round(v, 4) for k, v in self.signal_summary.items()},
            "triggered_words": self.triggered_words,
            "llm_summary": self.llm_summary,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PropagandaDetector:
    """
    High-level wrapper around a trained sklearn Pipeline.

    Parameters
    ----------
    model_name : filename (without extension) of the saved model in models/.
    threshold  : probability threshold above which text is labelled "Biased".
                 Default 0.5; lower to increase recall, raise for precision.
    """

    def __init__(self, model_name: str = "best_model", threshold: float = 0.5):
        self.threshold = threshold
        self._pipeline = load_pipeline(model_name)
        self._ling = LinguisticFeatureExtractor()

    # ------------------------------------------------------------------
    def predict(self, text: str) -> PredictionResult:
        """Classify a single text string."""
        texts = [text]
        proba = self._pipeline.predict_proba(texts)[0]
        p_biased = float(proba[1])
        p_non = float(proba[0])
        is_biased = p_biased >= self.threshold
        label = "Propaganda / Biased" if is_biased else "Non-biased"
        confidence = p_biased if is_biased else p_non

        # Linguistic feature breakdown
        ling_vec = self._ling._extract(text)
        signal_names = LinguisticFeatureExtractor.FEATURE_NAMES
        signal_summary = {
            name: float(ling_vec[i])
            for i, name in enumerate(signal_names)
            if name in (
                "vader_compound", "vader_neg", "loaded_word_ratio",
                "assertive_ratio", "caps_ratio", "exclamation_count",
            )
        }

        # Triggered words
        tokens = text.lower().split()
        triggered = list(
            {t for t in tokens if t in LOADED_WORDS | ASSERTIVE_WORDS}
        )

        return PredictionResult(
            text=text,
            label=label,
            confidence=confidence,
            probability_biased=p_biased,
            probability_nonbiased=p_non,
            signal_summary=signal_summary,
            triggered_words=triggered,
            llm_summary=None,  # Reserved for LLM integration
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Classify a list of texts efficiently."""
        return [self.predict(t) for t in texts]

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Classify all rows of a DataFrame.

        Adds columns: label, confidence, probability_biased, triggered_words.
        """
        results = self.predict_batch(df[text_col].tolist())
        df = df.copy()
        df["label"] = [r.label for r in results]
        df["confidence"] = [r.confidence for r in results]
        df["probability_biased"] = [r.probability_biased for r in results]
        df["triggered_words"] = [", ".join(r.triggered_words) for r in results]
        return df


# ---------------------------------------------------------------------------
# Transformer-based detector (DistilBERT)
# ---------------------------------------------------------------------------

class TransformerDetector:
    """
    Same predict() interface as PropagandaDetector, but backed by a
    fine-tuned DistilBERT model instead of a sklearn pipeline.
    """

    def __init__(self, model_name: str = "distilbert_model", threshold: float = 0.5):
        self.threshold = threshold
        from .transformer_model import DistilBertTrainer
        self._trainer = DistilBertTrainer.load(model_name)
        self._ling = LinguisticFeatureExtractor()

    def predict(self, text: str) -> PredictionResult:
        proba = self._trainer.predict_proba([text])[0]
        p_biased = float(proba[1])
        p_non = float(proba[0])
        is_biased = p_biased >= self.threshold
        label = "Propaganda / Biased" if is_biased else "Non-biased"
        confidence = p_biased if is_biased else p_non

        ling_vec = self._ling._extract(text)
        signal_summary = {
            name: float(ling_vec[i])
            for i, name in enumerate(LinguisticFeatureExtractor.FEATURE_NAMES)
            if name in (
                "vader_compound", "vader_neg", "loaded_word_ratio",
                "assertive_ratio", "caps_ratio", "exclamation_count",
            )
        }

        tokens = text.lower().split()
        triggered = list({t for t in tokens if t in LOADED_WORDS | ASSERTIVE_WORDS})

        return PredictionResult(
            text=text,
            label=label,
            confidence=confidence,
            probability_biased=p_biased,
            probability_nonbiased=p_non,
            signal_summary=signal_summary,
            triggered_words=triggered,
            llm_summary=None,
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        return [self.predict(t) for t in texts]

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        results = self.predict_batch(df[text_col].tolist())
        df = df.copy()
        df["label"] = [r.label for r in results]
        df["confidence"] = [r.confidence for r in results]
        df["probability_biased"] = [r.probability_biased for r in results]
        df["triggered_words"] = [", ".join(r.triggered_words) for r in results]
        return df


# ---------------------------------------------------------------------------
# Optional: scrape text from a URL (requires requests + bs4)
# ---------------------------------------------------------------------------

def fetch_text_from_url(url: str, timeout: int = 10) -> str:
    """
    Fetch the main text body from a URL for analysis.

    Requires: requests, beautifulsoup4  (not in requirements by default).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "URL fetching requires 'requests' and 'beautifulsoup4'.\n"
            "Install with: pip install requests beautifulsoup4"
        ) from exc

    headers = {"User-Agent": "Mozilla/5.0 (compatible; PropagandaDetector/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts / styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    paragraphs = [p.get_text(separator=" ").strip() for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) > 40)
    return text
