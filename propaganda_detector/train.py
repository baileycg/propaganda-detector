"""
train.py – Train a propaganda-detection model on the MBIC dataset.

Usage
-----
    # Basic (auto-selects best model)
    python train.py --data data/raw/mbic.csv

    # Choose a specific classifier
    python train.py --data data/raw/mbic.csv --classifier logistic

    # Compare all classifiers and save the best
    python train.py --data data/raw/mbic.csv --compare

    # Evaluate on the held-out test set after training
    python train.py --data data/raw/mbic.csv --compare --final-eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_mbic, split_dataset, save_splits
from src.models import (
    build_pipeline,
    train,
    evaluate,
    compare_models,
    save_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a propaganda / media-bias detection model."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the MBIC CSV file (e.g. data/raw/mbic.csv).",
    )
    parser.add_argument(
        "--classifier",
        default="logistic",
        choices=["logistic", "svm", "rf", "gradient", "voting"],
        help="Classifier to train (ignored when --compare is set).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train all classifiers and pick the best by validation F1.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Evaluate the best model on the held-out test set.",
    )
    parser.add_argument(
        "--tfidf-features",
        type=int,
        default=15_000,
        help="Maximum TF-IDF vocabulary size (default: 15000).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data for the test set (default: 0.15).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of data for the validation set (default: 0.15).",
    )
    parser.add_argument(
        "--model-name",
        default="best_model",
        help="Filename (no extension) used to save the model (default: best_model).",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for the text in the CSV (default: text).",
    )
    parser.add_argument(
        "--label-col",
        default="label_bias",
        help="Column name for the label in the CSV (default: label_bias).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load dataset
    logger.info("Loading dataset from: %s", args.data)
    df = load_mbic(
        args.data,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    # 2. Split
    train_df, val_df, test_df = split_dataset(
        df, test_size=args.test_size, val_size=args.val_size
    )
    save_splits(train_df, val_df, test_df)

    # 3. Train
    if args.compare:
        best_name, best_pipeline, results = compare_models(
            train_texts=train_df["text"],
            train_labels=train_df["label"],
            val_texts=val_df["text"],
            val_labels=val_df["label"],
            max_tfidf_features=args.tfidf_features,
        )
        logger.info("Selected model: %s", best_name)

        # Print comparison table
        print("\n=== Model comparison (validation set) ===")
        print(f"{'Model':<12} {'Accuracy':>9} {'F1':>8} {'AUC':>8}")
        print("-" * 42)
        for name, m in results.items():
            print(f"{name:<12} {m['accuracy']:>9.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
        print()

        pipeline = best_pipeline
    else:
        pipeline = build_pipeline(
            classifier_name=args.classifier,
            max_tfidf_features=args.tfidf_features,
        )
        pipeline = train(pipeline, train_df["text"], train_df["label"])
        evaluate(pipeline, val_df["text"], val_df["label"], split_name="validation")

    # 4. Final test-set evaluation
    if args.final_eval:
        logger.info("Running final evaluation on held-out test set…")
        metrics = evaluate(pipeline, test_df["text"], test_df["label"], split_name="test")
        print("\n=== FINAL TEST RESULTS ===")
        print(metrics["report"])

    # 5. Save
    save_path = save_pipeline(pipeline, name=args.model_name)
    print(f"\nModel saved to: {save_path}")
    print("Run `python main.py --text <your text>` to classify new content.")


if __name__ == "__main__":
    main()
