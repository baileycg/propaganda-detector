"""
Data loader for the Media Bias Group's MBIC dataset.

Dataset source:
  https://github.com/Media-Bias-Group/media-bias-identification-corpus

The MBIC corpus labels news sentences for media bias (Biased / Non-biased).
We treat "Biased" as a proxy for propagandistic / manipulative language.

Expected CSV columns (MBIC format):
  - sentence      : the text fragment to classify
  - label_bias    : "Biased" | "Non-biased" | "No agreement"
  - label_opinion : opinion label (optional, used for stratification)
  - url           : source article URL (optional)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Column names in the MBIC dataset
TEXT_COL = "text"
LABEL_COL = "label_bias"

# Labels
BIASED_LABEL = "Biased"
NON_BIASED_LABEL = "Non-biased"
NO_AGREEMENT = "No agreement"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_mbic(
    csv_path: str | Path,
    drop_no_agreement: bool = True,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Load and normalise the MBIC CSV file.

    Parameters
    ----------
    csv_path : path to the MBIC CSV file.
    drop_no_agreement : if True, rows where annotators did not agree are removed.
    text_col : column that contains the sentence text.
    label_col : column that contains the bias label.

    Returns
    -------
    DataFrame with columns ['text', 'label'] where label is 0 (non-biased) or
    1 (biased / propagandistic).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Download it from: https://github.com/Media-Bias-Group/media-bias-identification-corpus\n"
            "and place the CSV file in propaganda_detector/data/raw/"
        )

    logger.info("Loading dataset from %s", csv_path)

    # MBIC files are semicolon-separated with quoted multi-line article fields
    try:
        df = pd.read_csv(csv_path, sep=";", quotechar='"', on_bad_lines="skip", engine="python")
    except Exception as exc:
        raise RuntimeError(f"Could not parse {csv_path}: {exc}") from exc

    # Validate required columns
    for col in (text_col, label_col):
        if col not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Expected column '{col}' not found.\n"
                f"Available columns: {available}\n"
                "Set text_col / label_col to the correct names."
            )

    df = df[[text_col, label_col]].copy()
    df.rename(columns={text_col: "text", label_col: "raw_label"}, inplace=True)

    # Drop rows with missing text
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    if drop_no_agreement:
        before = len(df)
        df = df[df["raw_label"] != NO_AGREEMENT]
        removed = before - len(df)
        if removed:
            logger.info("Removed %d 'No agreement' rows.", removed)

    # Binary label: 1 = biased/propagandistic, 0 = non-biased
    df["label"] = (df["raw_label"] == BIASED_LABEL).astype(int)
    df.drop(columns=["raw_label"], inplace=True)

    logger.info(
        "Loaded %d samples | biased=%d (%.1f%%) | non-biased=%d (%.1f%%)",
        len(df),
        df["label"].sum(),
        100 * df["label"].mean(),
        (df["label"] == 0).sum(),
        100 * (1 - df["label"].mean()),
    )

    return df.reset_index(drop=True)


def load_custom(
    csv_path: str | Path,
    text_col: str,
    label_col: str,
    positive_label: str,
) -> pd.DataFrame:
    """
    Generic loader for any CSV dataset.

    Parameters
    ----------
    csv_path : path to CSV file.
    text_col : name of the text column.
    label_col : name of the label column.
    positive_label : the string value in label_col that means "biased/propaganda".

    Returns
    -------
    DataFrame with columns ['text', 'label'].
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df[[text_col, label_col]].copy()
    df.rename(columns={text_col: "text", label_col: "raw_label"}, inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = (df["raw_label"] == positive_label).astype(int)
    df.drop(columns=["raw_label"], inplace=True)
    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val, stratify=temp_df["label"], random_state=random_state
    )
    logger.info(
        "Split | train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    logger.info("Saved processed splits to %s", out_dir)


def load_splits(out_dir: Path = PROCESSED_DIR):
    return (
        pd.read_csv(out_dir / "train.csv"),
        pd.read_csv(out_dir / "val.csv"),
        pd.read_csv(out_dir / "test.csv"),
    )
