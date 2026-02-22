"""
src/emotion_analyzer.py

Two-layer emotion analysis for detected propaganda text.

  Layer 1 (word-level)     : NRC Emotion Lexicon via nrclex
                             Returns normalised scores for 8 emotions.
  Layer 2 (sentence-level) : j-hartmann/emotion-english-distilroberta-base
                             Pre-trained transformer, lazy-loaded on first use.

Usage
-----
    from src.emotion_analyzer import EmotionAnalyzer
    ea = EmotionAnalyzer()
    scores = ea.analyze("The corrupt regime destroyed everything!!!", run_hf=True)
    # {"nrc_anger": 0.0833, "nrc_fear": 0.0417, ..., "hf_anger": 0.4821, ...}
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

NRC_EMOTIONS = [
    "anger", "fear", "anticipation", "trust",
    "surprise", "sadness", "disgust", "joy",
]

HF_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


class EmotionAnalyzer:
    """
    Analyzes the emotional content of a text using two complementary methods.

    NRC layer  : word-level, always available, no GPU needed.
    HF layer   : sentence-level DistilRoBERTa, GPU-accelerated, lazy-loaded.
                 Only instantiated on the first call to analyze_hf().
    """

    def __init__(self) -> None:
        self._hf_pipeline = None  # loaded lazily on first HF call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_nrc(self, text: str) -> Dict[str, float]:
        """
        Return normalised NRC emotion scores for the text.

        Scores are word-count proportions (0.0 - 1.0).  All 8 NRC emotion
        keys are always present, even if zero.
        """
        try:
            from nrclex import NRCLex
        except ImportError as exc:
            raise ImportError(
                "nrclex is required for NRC emotion analysis.\n"
                "Install with: py -3 -m pip install nrclex"
            ) from exc

        emotion_obj = NRCLex(text)
        raw: dict = emotion_obj.raw_emotion_scores  # e.g. {"anger": 3, "joy": 1}

        word_count = max(len(text.split()), 1)
        return {
            emotion: round(raw.get(emotion, 0) / word_count, 4)
            for emotion in NRC_EMOTIONS
        }

    def analyze_hf(self, text: str) -> Dict[str, float]:
        """
        Return HuggingFace DistilRoBERTa emotion probabilities.

        Lazily loads the pipeline on first call (GPU if available).
        Returns all 7 emotion labels and their probabilities.
        """
        self._ensure_hf_loaded()
        truncated = text[:512]  # avoid token-length errors on long texts
        result = self._hf_pipeline(truncated, top_k=None)
        return {item["label"]: round(float(item["score"]), 4) for item in result}

    def analyze(self, text: str, run_hf: bool = False) -> Dict[str, float]:
        """
        Merge both layers into a single flat dict.

        NRC keys are prefixed with "nrc_".
        HF  keys are prefixed with "hf_".
        If run_hf is False, only NRC scores are returned.
        """
        emotions: Dict[str, float] = {}

        for k, v in self.analyze_nrc(text).items():
            emotions[f"nrc_{k}"] = v

        if run_hf:
            try:
                for k, v in self.analyze_hf(text).items():
                    emotions[f"hf_{k}"] = v
            except Exception as exc:
                logger.warning("HF emotion analysis failed: %s", exc)

        return emotions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_hf_loaded(self) -> None:
        """Load the HuggingFace pipeline exactly once."""
        if self._hf_pipeline is not None:
            return
        try:
            from transformers import pipeline
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for HF emotion analysis."
            ) from exc

        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading HF emotion model '%s' (device=%d) ...", HF_EMOTION_MODEL, device
        )
        self._hf_pipeline = pipeline(
            "text-classification",
            model=HF_EMOTION_MODEL,
            top_k=None,
            device=device,
        )
        logger.info("HF emotion model ready.")
