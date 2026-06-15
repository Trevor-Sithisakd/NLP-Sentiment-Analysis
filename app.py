"""
CLI entry point for the sentiment analyser.

Usage:
    python app.py                  # interactive mode
    python app.py messages.txt     # batch file mode
"""

import sys
import logging

from predictor import SentimentPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_interactive_mode(predictor: SentimentPredictor) -> None:
    """
    Run a REPL loop that reads text from stdin and prints sentiment predictions.
    Exits when the user types 'exit' or 'quit'.
    """
    print("\nTwitter-RoBERTa Sentiment Analyser")
    print("Type your text (or 'exit' to quit):\n")

    while True:
        user_input = input("Your text: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        label, score = predictor.predict(user_input)
        print(f"→ Sentiment: {label}  (Confidence: {score:.2f})\n")


def run_file_mode(predictor: SentimentPredictor, filepath: str) -> None:
    """
    Read lines from a text file and print sentiment predictions for each.

    Each non-empty line is treated as a separate text to analyse.
    """
    try:
        with open(filepath, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        sys.exit(1)
    except OSError as exc:
        logger.error("Could not read file '%s': %s", filepath, exc)
        sys.exit(1)

    if not lines:
        print("File is empty — nothing to analyse.")
        return

    results = predictor.predict_batch(lines)

    print(f"\nBatch Sentiment Results ({len(lines)} lines):\n")
    for text, (label, score) in zip(lines, results):
        print(f"  {text}")
        print(f"  → {label} (Confidence: {score:.2f})\n")


def main() -> None:
    predictor = SentimentPredictor()

    if len(sys.argv) > 1:
        run_file_mode(predictor, filepath=sys.argv[1])
    else:
        run_interactive_mode(predictor)


if __name__ == "__main__":
    main()
