"""
ptc_loader.py – Loader for the Propaganda Techniques Corpus (PTC) v2.

SemEval 2020 Task 11 dataset structure (datasets/ directory):
  train-articles/               371 plain-text article files
  train-labels-task1-span-identification/   per-article span labels
  train-task1-SI.labels         combined span labels for all train articles

Each article file has one sentence per line (newline-separated).
The combined label file is tab-separated:
  article_id  start_offset  end_offset

We convert article-level character-span annotations to sentence-level binary
labels: a sentence is labelled 1 (propaganda) if ANY propaganda span overlaps
its character range, 0 otherwise.

Usage
-----
    from src.ptc_loader import load_ptc
    df = load_ptc("C:/Users/Bailey/Downloads/datasets-v2/datasets")
    # df has columns ['text', 'label']  — same schema as MBIC output
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_ptc(
    ptc_dir: str | Path,
    min_sentence_len: int = 20,
) -> pd.DataFrame:
    """
    Load PTC v2 training data as sentence-level binary labels.

    Parameters
    ----------
    ptc_dir : path to the datasets/ directory that contains
              ``train-articles/`` and ``train-task1-SI.labels``.
    min_sentence_len : sentences shorter than this (chars) are skipped —
                       catches blank lines and very short headings.

    Returns
    -------
    DataFrame with columns ['text', 'label']
    where label=1 means the sentence contains propaganda and label=0 means it
    does not.  Compatible with the MBIC output schema.
    """
    ptc_dir = Path(ptc_dir)
    articles_dir = ptc_dir / "train-articles"
    labels_file = ptc_dir / "train-task1-SI.labels"

    if not articles_dir.exists():
        raise FileNotFoundError(
            f"train-articles/ directory not found in {ptc_dir}.\n"
            "Make sure ptc_dir points to the datasets/ folder."
        )
    if not labels_file.exists():
        raise FileNotFoundError(
            f"train-task1-SI.labels not found in {ptc_dir}.\n"
            "Make sure ptc_dir points to the datasets/ folder."
        )

    # ------------------------------------------------------------------
    # 1. Load all propaganda spans into a dict keyed by article_id
    # ------------------------------------------------------------------
    spans: Dict[str, List[Tuple[int, int]]] = {}
    with open(labels_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                logger.debug("Skipping malformed label line: %r", line)
                continue
            article_id, start, end = parts[0], int(parts[1]), int(parts[2])
            spans.setdefault(article_id, []).append((start, end))

    logger.info(
        "Loaded propaganda spans for %d articles from %s", len(spans), labels_file.name
    )

    # ------------------------------------------------------------------
    # 2. Convert each article to sentence-level labels
    # ------------------------------------------------------------------
    rows = []
    article_files = sorted(articles_dir.glob("article*.txt"))
    logger.info("Processing %d training articles…", len(article_files))

    for article_file in article_files:
        # Article id is the numeric part after "article"
        article_id = article_file.stem[len("article"):]

        try:
            full_text = article_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            full_text = article_file.read_text(encoding="latin-1")

        article_spans = spans.get(article_id, [])

        # Walk lines, tracking cumulative character offset
        offset = 0
        for raw_line in full_text.split("\n"):
            line_start = offset
            line_end = offset + len(raw_line)
            offset = line_end + 1  # +1 for the stripped newline character

            sentence = raw_line.strip()
            if len(sentence) < min_sentence_len:
                continue

            # A sentence is "propaganda" if any span overlaps its range.
            # Overlap condition: span_start < line_end  AND  span_end > line_start
            is_propaganda = any(
                s < line_end and e > line_start
                for s, e in article_spans
            )
            rows.append({"text": sentence, "label": int(is_propaganda)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No sentences were extracted from the PTC dataset. "
            "Check that ptc_dir contains the correct files."
        )

    n_prop = int(df["label"].sum())
    n_non  = len(df) - n_prop
    logger.info(
        "PTC loaded | total=%d | propaganda=%d (%.1f%%) | non-propaganda=%d (%.1f%%)",
        len(df),
        n_prop,
        100 * n_prop / len(df),
        n_non,
        100 * n_non / len(df),
    )

    return df.reset_index(drop=True)
