"""
Sentiment prediction using a Twitter-RoBERTa transformer model.

Model loading is deferred until first use (lazy initialisation) so that
import errors are surfaced clearly and startup time is minimised.
"""

import logging
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from cleaning import TextCleaner

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Maps raw model output labels to human-readable sentiment names.
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

SentimentResult = Tuple[str, float]  # (label, confidence_score)


class SentimentPredictor:
    """
    Loads a transformer sentiment model and exposes prediction methods.

    The model is loaded once on first use. All text is cleaned before
    inference using TextCleaner.

    Example:
        predictor = SentimentPredictor()
        label, score = predictor.predict("I love this so much!!")
    """

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self._cleaner = cleaner or TextCleaner()
        self._classifier = None  # Lazy-loaded on first predict call

    def _load_model(self) -> None:
        """Load tokenizer, model, and pipeline. Called once on first use."""
        logger.info("Loading sentiment model: %s", MODEL_NAME)
        try:
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self._classifier = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            logger.info("Model loaded successfully on device=%d", device)
        except Exception as exc:
            logger.error("Failed to load sentiment model '%s': %s", MODEL_NAME, exc)
            raise RuntimeError(
                f"Could not load model '{MODEL_NAME}'. "
                "Check your internet connection and Hugging Face access."
            ) from exc

    @property
    def classifier(self):
        """Return the classifier, loading it on first access."""
        if self._classifier is None:
            self._load_model()
        return self._classifier

    def _map_label(self, raw_label: str) -> str:
        """Convert a raw model label to a human-readable sentiment string."""
        mapped = LABEL_MAP.get(raw_label)
        if mapped is None:
            logger.warning("Unexpected model label '%s'; returning as-is.", raw_label)
            return raw_label
        return mapped

    def predict(self, text: str) -> SentimentResult:
        """
        Predict the sentiment of a single text.

        Returns ("Empty Input", 0.0) if the text is empty after cleaning.
        """
        cleaned_text = self._cleaner.clean(text)
        if not cleaned_text:
            return ("Empty Input", 0.0)

        result = self.classifier(cleaned_text)[0]
        return self._map_label(result["label"]), float(result["score"])

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Predict sentiment for a list of texts.

        Texts that are empty after cleaning are returned as ("Empty Input", 0.0)
        without being sent to the model.
        """
        cleaned_texts = self._cleaner.clean_batch(texts)

        results: List[SentimentResult] = []
        for original_text, cleaned_text in zip(texts, cleaned_texts):
            if not cleaned_text:
                logger.debug("Empty input after cleaning: %r", original_text)
                results.append(("Empty Input", 0.0))
            else:
                raw = self.classifier(cleaned_text)[0]
                results.append((self._map_label(raw["label"]), float(raw["score"])))

        return results
