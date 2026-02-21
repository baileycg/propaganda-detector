"""
train_transformer.py – Fine-tune DistilBERT for propaganda / bias detection.

Usage
-----
    # Basic (CPU, ~15-25 min)
    python train_transformer.py --data data/raw/mbic.csv

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
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for propaganda / media-bias detection."
    )
    parser.add_argument("--data", required=True, help="Path to MBIC CSV file.")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length (default: 128).")
    parser.add_argument("--model-name", default="distilbert_model", help="Name to save model under.")
    parser.add_argument("--use-existing-splits", action="store_true",
                        help="Use previously saved train/val/test splits instead of re-splitting.")
    parser.add_argument("--final-eval", action="store_true",
                        help="Evaluate on held-out test set after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load data
    if args.use_existing_splits and (PROCESSED_DIR / "train.csv").exists():
        logger.info("Loading existing splits from %s", PROCESSED_DIR)
        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
        test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    else:
        logger.info("Loading and splitting dataset from %s", args.data)
        df = load_mbic(args.data)
        train_df, val_df, test_df = split_dataset(df)
        save_splits(train_df, val_df, test_df)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts   = val_df["text"].tolist()
    val_labels  = val_df["label"].tolist()

    logger.info(
        "Dataset | train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df)
    )

    # 2. Train
    trainer = DistilBertTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )
    trainer.fit(train_texts, train_labels, val_texts, val_labels)

    # 3. Optional final test eval
    if args.final_eval:
        trainer.evaluate(
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            split_name="test",
        )

    # 4. Save
    trainer.save(name=args.model_name)
    print(f"\nDone. Run predictions with:")
    print(f"  PYTHONUTF8=1 py -3 main.py --model-type transformer --text \"your text here\"")


if __name__ == "__main__":
    main()
