"""
Text pre-processing utilities for Twitter/TikTok sentiment analysis.

All cleaning logic is encapsulated in TextCleaner to make dependencies
explicit and allow easy substitution or testing.
"""

import re
import string
import logging
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

_NLTK_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]


def _download_nltk_resources() -> None:
    """Download required NLTK data files if not already present."""
    for resource in _NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


class TextCleaner:
    """
    Cleans raw social-media text for use with transformer sentiment models.

    Removes URLs, mentions, hashtags, punctuation, digits, and stopwords,
    then lemmatizes the remaining tokens.

    Example:
        cleaner = TextCleaner()
        cleaned = cleaner.clean("Check out https://example.com @user #vibes!!")
    """

    _URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    _MENTION_HASHTAG_PATTERN = re.compile(r"[@#][\w_]+")
    _WHITESPACE_PATTERN = re.compile(r"\s+")
    _STRIP_CHARS = str.maketrans("", "", string.punctuation + string.digits)

    def __init__(self) -> None:
        _download_nltk_resources()
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """
        Clean a single text string.

        Returns an empty string if the input contains no meaningful tokens
        after cleaning.
        """
        text = text.lower()
        text = self._URL_PATTERN.sub("", text)
        text = self._MENTION_HASHTAG_PATTERN.sub("", text)
        text = text.translate(self._STRIP_CHARS)
        text = self._WHITESPACE_PATTERN.sub(" ", text).strip()

        tokens = [
            self._lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self._stop_words
        ]
        return " ".join(tokens)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of text strings."""
        return [self.clean(t) for t in texts]
