"""
main.py – CLI for the propaganda / media-bias detector.

Examples
--------
    # Classify a single text string
    python main.py --text "The corrupt regime destroyed everything we worked for!!!"

    # Classify text from a URL (requires requests + beautifulsoup4)
    python main.py --url https://example.com/some-article

    # Classify each line of a CSV
    python main.py --csv data/my_articles.csv --text-col content --output results.csv

    # Interactive mode
    python main.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import PropagandaDetector, TransformerDetector, fetch_text_from_url

logging.basicConfig(
    level=logging.WARNING,  # quieter for end-user CLI
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect propaganda / media bias in text or web content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "The corrupt regime destroyed everything we worked for!!!"
  python main.py --url https://example.com/article
  python main.py --csv articles.csv --text-col body --output results.csv
  python main.py --interactive
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw text string to classify.")
    input_group.add_argument("--url", help="URL to fetch and classify.")
    input_group.add_argument("--csv", help="Path to a CSV file for batch classification.")
    input_group.add_argument(
        "--interactive", action="store_true", help="Enter interactive classification mode."
    )

    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for text when using --csv (default: text).",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path for batch results (default: prints to stdout).",
    )
    parser.add_argument(
        "--model",
        default="best_model",
        help="Saved model name to load (default: best_model / distilbert_model).",
    )
    parser.add_argument(
        "--model-type",
        default="sklearn",
        choices=["sklearn", "transformer"],
        help="Which model backend to use: 'sklearn' (default, fast) or 'transformer' (DistilBERT, more accurate).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for 'Biased' label (default: 0.5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of formatted text.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_text(detector: PropagandaDetector, text: str, as_json: bool) -> None:
    result = detector.predict(text)
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)


def handle_url(detector: PropagandaDetector, url: str, as_json: bool) -> None:
    print(f"Fetching: {url}")
    try:
        text = fetch_text_from_url(url)
    except Exception as exc:
        print(f"Error fetching URL: {exc}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Warning: No text extracted from page.", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(text)} characters of text.\n")

    # Classify the full body as one block
    result = detector.predict(text)
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)


def handle_csv(
    detector: PropagandaDetector,
    csv_path: str,
    text_col: str,
    output_path: str | None,
    as_json: bool,
) -> None:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        print(
            f"Column '{text_col}' not found in {csv_path}.\n"
            f"Available: {', '.join(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Classifying {len(df)} rows…")
    out_df = detector.predict_dataframe(df, text_col=text_col)

    biased_count = (out_df["label"] == "Propaganda / Biased").sum()
    print(f"Found {biased_count}/{len(df)} potentially biased entries ({100*biased_count/len(df):.1f}%)")

    if output_path:
        out_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        if as_json:
            print(out_df[["text" if text_col == "text" else text_col, "label", "confidence", "probability_biased"]].to_json(orient="records", indent=2))
        else:
            pd.set_option("display.max_colwidth", 60)
            print(out_df[["label", "confidence", "probability_biased", "triggered_words"]].to_string())


def handle_interactive(detector: PropagandaDetector) -> None:
    print("=== Propaganda Detector – Interactive Mode ===")
    print("Type or paste text to classify. Enter 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("Text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        result = detector.predict(text)
        print(result)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Default model name differs by backend
    model_name = args.model
    if model_name == "best_model" and args.model_type == "transformer":
        model_name = "distilbert_model"

    try:
        if args.model_type == "transformer":
            detector = TransformerDetector(model_name=model_name, threshold=args.threshold)
        else:
            detector = PropagandaDetector(model_name=model_name, threshold=args.threshold)
    except FileNotFoundError as exc:
        print(f"Model not found: {exc}", file=sys.stderr)
        if args.model_type == "transformer":
            print("Train one with: python train_transformer.py --data data/raw/mbic.csv", file=sys.stderr)
        else:
            print("Train one with: python train.py --data data/raw/mbic.csv", file=sys.stderr)
        sys.exit(1)

    if args.text:
        handle_text(detector, args.text, as_json=args.json)
    elif args.url:
        handle_url(detector, args.url, as_json=args.json)
    elif args.csv:
        handle_csv(detector, args.csv, args.text_col, args.output, as_json=args.json)
    elif args.interactive:
        handle_interactive(detector)


if __name__ == "__main__":
    main()
