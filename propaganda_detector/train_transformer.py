"""
train_transformer.py – Fine-tune DistilBERT for propaganda / bias detection.

Usage
-----
    # Basic – MBIC only (CPU, ~15-25 min)
    python train_transformer.py --data data/raw/mbic.csv

    # With PTC v2 for better generalisation (recommended)
    python train_transformer.py --data data/raw/mbic.csv --ptc-dir "C:/Users/Bailey/Downloads/datasets-v2/datasets"

    # Fewer epochs for a quick test
    python train_transformer.py --data data/raw/mbic.csv --epochs 2

    # Custom model save name
    python train_transformer.py --data data/raw/mbic.csv --model-name my_distilbert
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_mbic, split_dataset, save_splits, PROCESSED_DIR
from src.transformer_model import DistilBertTrainer

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for propaganda / media-bias detection."
    )
    parser.add_argument("--data", required=True, help="Path to MBIC CSV file.")
    parser.add_argument(
        "--ptc-dir",
        default=None,
        help=(
            "Path to the PTC v2 datasets/ directory "
            "(e.g. C:/Users/.../datasets-v2/datasets). "
            "When provided, PTC sentences are merged into training data."
        ),
    )
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length (default: 128).")
    parser.add_argument("--model-name", default="distilbert_model", help="Name to save model under.")
    parser.add_argument(
        "--use-existing-splits",
        action="store_true",
        help="Use previously saved train/val/test splits instead of re-splitting.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Evaluate on held-out test set after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load MBIC and split into train / val / test
    # ------------------------------------------------------------------
    if args.use_existing_splits and (PROCESSED_DIR / "train.csv").exists():
        logger.info("Loading existing splits from %s", PROCESSED_DIR)
        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
        test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    else:
        logger.info("Loading and splitting MBIC dataset from %s", args.data)
        df = load_mbic(args.data)
        train_df, val_df, test_df = split_dataset(df)
        save_splits(train_df, val_df, test_df)

    logger.info(
        "MBIC split | train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ------------------------------------------------------------------
    # 2. Optionally augment training data with PTC v2
    # ------------------------------------------------------------------
    if args.ptc_dir:
        from src.ptc_loader import load_ptc

        logger.info("Loading PTC v2 dataset from %s", args.ptc_dir)
        ptc_df = load_ptc(args.ptc_dir)

        # Split PTC into a small held-out val slice and a large train slice.
        # We keep the MBIC test set untouched for a clean eval on that domain.
        from sklearn.model_selection import train_test_split
        ptc_train, ptc_val = train_test_split(
            ptc_df, test_size=0.10, stratify=ptc_df["label"], random_state=42
        )

        before = len(train_df)
        train_df = pd.concat([train_df, ptc_train], ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        val_df = pd.concat([val_df, ptc_val], ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

        logger.info(
            "After PTC augmentation | train=%d (+%d)  val=%d",
            len(train_df), len(train_df) - before, len(val_df),
        )
    else:
        logger.info("No --ptc-dir given; training on MBIC only.")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts   = val_df["text"].tolist()
    val_labels  = val_df["label"].tolist()

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    trainer = DistilBertTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )
    trainer.fit(train_texts, train_labels, val_texts, val_labels)

    # ------------------------------------------------------------------
    # 4. Optional final test eval (MBIC test set — domain-consistent)
    # ------------------------------------------------------------------
    if args.final_eval:
        trainer.evaluate(
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            split_name="test",
        )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    trainer.save(name=args.model_name)
    print("\nDone. Run predictions with:")
    print("  py -3 main.py --model-type transformer --text \"your text here\"")


if __name__ == "__main__":
    main()
